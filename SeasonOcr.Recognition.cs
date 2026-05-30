// Copyright (c) SeasonRealms and contributors.
// Licensed under the Apache License, Version 2.0.
// SeasonOCR for EasyOCR Models
//
// This file contains a C# translation/adaptation of logic from the EasyOCR project:
// https://github.com/JaidedAI/EasyOCR
// Original project license: Apache-2.0.
// Portions of the detection pipeline trace back to CRAFT-related code distributed under the MIT License.
// Modified from the original Python implementation.

namespace SeasonOCR;

public static partial class SeasonOcr
{
    /// <summary>
    /// Builds a fallback charset when the ONNX model does not embed one in its metadata.
    /// A 352-class model is interpreted as blank plus 95 ASCII, 128 Latin-1, and 128 Latin Extended-A characters.
    /// </summary>
    static string BuildFallbackCharset(InferenceSession session, string outputName)
    {
        var outMeta = session.OutputMetadata[outputName];
        int numChars = 95; // Default printable ASCII count.
        if (outMeta.Dimensions.Length >= 3 && outMeta.Dimensions[2] > 0)
            numChars = outMeta.Dimensions[2] - 1;
        else if (outMeta.Dimensions.Length >= 2 && outMeta.Dimensions[1] > 0)
            numChars = outMeta.Dimensions[1] - 1;

        // Build the Latin OCR charset using the PaddleOCR ordering.
        var sb = new StringBuilder();
        // 0-9
        for (char c = '0'; c <= '9'; c++) sb.Append(c);
        // a-z
        for (char c = 'a'; c <= 'z'; c++) sb.Append(c);
        // A-Z
        for (char c = 'A'; c <= 'Z'; c++) sb.Append(c);
        // PaddleOCR punctuation ordering, with space at the end.
        sb.Append("!\"#$%&'()*+,-./");
        sb.Append(":;<=>?@");
        sb.Append("[\\]^_`");
        sb.Append("{|}~ ");

        if (numChars > 96)
        {
            // Latin-1 Supplement: U+00A1 to U+00FF in Unicode code-point order.
            for (int cp = 0x00A1; cp <= 0x00FF && sb.Length < numChars; cp++)
                sb.Append((char)cp);
        }
        if (numChars > 224)
        {
            // Latin Extended-A: U+0100 to U+017F.
            for (int cp = 0x0100; cp <= 0x017F && sb.Length < numChars; cp++)
                sb.Append((char)cp);
        }

        // Pad with spaces if the inferred charset is still too short.
        while (sb.Length < numChars)
            sb.Append(' ');
        return sb.ToString().Substring(0, numChars);
    }

    static DecoderLexicon LoadDecoderLexicon(
        IReadOnlyDictionary<string, string> metadata,
        bool enableWordBeamSearch,
        string? dictionary,
        string charset,
        IReadOnlyList<string> languages)
    {
        if (!enableWordBeamSearch)
            return new DecoderLexicon();

        var lexicon = new DecoderLexicon();
        if (!string.IsNullOrWhiteSpace(dictionary))
        {
            LoadWordsIntoSet(dictionary, lexicon.CombinedWords);
            DebugLog($"[SeasonOCR] wordbeamsearch custom dictionary loaded: {dictionary.Length} ({lexicon.CombinedWords.Count} words)");
            return lexicon;
        }

        DebugLog($"[SeasonOCR] wordbeamsearch languages=[{string.Join(",", languages)}]");
        bool loadedFromMetadata = LoadEmbeddedDictionaries(metadata, languages, lexicon);
        if (loadedFromMetadata)
            DebugLog("[SeasonOCR] wordbeamsearch dictionaries loaded from model metadata.");

        foreach (var (lang, separators) in GetDefaultSeparatorCharMap())
        {
            if (!charset.Contains(separators.Start) || !charset.Contains(separators.End))
                continue;
            if (!lexicon.WordsByLanguage.ContainsKey(lang))
                continue;
            lexicon.SeparatorsByLanguage[lang] = separators;
            DebugLog($"[SeasonOCR] separator enabled: {lang} -> U+{(int)separators.Start:X4}/U+{(int)separators.End:X4}");
        }

        if (lexicon.CombinedWords.Count == 0)
            DebugLog("[SeasonOCR] wordbeamsearch dictionaries not found, fallback to beamsearch.");

        return lexicon;
    }

    static bool LoadEmbeddedDictionaries(
        IReadOnlyDictionary<string, string> metadata,
        IReadOnlyList<string> languages,
        DecoderLexicon lexicon)
    {
        if (metadata.Count == 0)
            return false;

        bool loadedAny = false;
        foreach (string lang in languages)
        {
            var key = $"dict_{lang}";

            if (!metadata.TryGetValue(key, out string? value) || string.IsNullOrWhiteSpace(value))
                continue;

            if (!lexicon.WordsByLanguage.TryGetValue(lang, out var words))
            {
                words = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                lexicon.WordsByLanguage[lang] = words;
            }

            int before = words.Count;
            LoadWordsIntoSet(ParseDictionaryEntries(value), words);
            if (words.Count == before)
                continue;

            foreach (string word in words)
                lexicon.CombinedWords.Add(word);

            loadedAny = true;
            DebugLog($"[SeasonOCR] wordbeamsearch embedded dictionary loaded: {lang} -> {key} ({words.Count} words)");
        }

        return loadedAny;
    }

    static IReadOnlyList<string> ParseLanguageMetadata(IReadOnlyDictionary<string, string> metadata)
    {
        if (metadata.Count == 0)
            return [];

        foreach (string key in new[] { "lang_list", "languages", "language_list" })
        {
            if (!metadata.TryGetValue(key, out string? value) || string.IsNullOrWhiteSpace(value))
                continue;

            var parsed = ParseLanguageListString(value);
            if (parsed.Count > 0)
                return parsed;
        }

        if (metadata.TryGetValue("model_lang", out string? modelLang) && !string.IsNullOrWhiteSpace(modelLang))
        {
            var parsed = InferModelLanguages(modelLang, modelLang).ToList();
            if (parsed.Count > 0)
                return parsed;
        }

        return [];
    }

    static IReadOnlyList<string> ParseLanguageListString(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return [];

        value = value.Trim();
        if (value.StartsWith("[", StringComparison.Ordinal))
        {
            try
            {
                var parsed = System.Text.Json.JsonSerializer.Deserialize<string[]>(value);
                return NormalizeLanguageList(parsed);
            }
            catch (System.Text.Json.JsonException)
            {
            }
        }

        return NormalizeLanguageList(
            value.Split([',', ';', ' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries));
    }

    static IReadOnlyList<string> NormalizeLanguageList(IEnumerable<string>? langList)
    {
        if (langList == null)
            return [];

        return langList
            .Where(lang => !string.IsNullOrWhiteSpace(lang))
            .Select(lang => lang.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    static IEnumerable<string> ParseDictionaryEntries(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            yield break;

        value = value.Trim();
        if (value.StartsWith("[", StringComparison.Ordinal))
        {
            string[]? parsed = null;
            try
            {
                parsed = System.Text.Json.JsonSerializer.Deserialize<string[]>(value);
            }
            catch (System.Text.Json.JsonException)
            {
            }

            if (parsed != null)
            {
                foreach (string item in parsed)
                {
                    if (!string.IsNullOrWhiteSpace(item))
                        yield return item.Trim();
                }
                yield break;
            }
        }

        foreach (string entry in value.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries))
        {
            string word = entry.Trim();
            if (!string.IsNullOrWhiteSpace(word))
                yield return word;
        }
    }

    static void LoadWordsIntoSet(IEnumerable<string> words, HashSet<string> target)
    {
        foreach (string word in words)
        {
            string trimmed = word.Trim();
            if (!string.IsNullOrWhiteSpace(trimmed))
                target.Add(trimmed);
        }
    }

    static void LoadWordsIntoSet(string dictionary, HashSet<string> target)
    {
        LoadWordsIntoSet(ParseDictionaryEntries(dictionary), target);
    }

    static IEnumerable<string> InferModelLanguages(string recognizerModel, string modelStem)
    {
        string stem = (Path.GetFileNameWithoutExtension(recognizerModel) ?? modelStem).ToLowerInvariant();
        if (stem.StartsWith("recognizer_"))
            stem = stem["recognizer_".Length..];

        var parts = stem.Split(['-', '_'], StringSplitOptions.RemoveEmptyEntries);
        bool HasToken(params string[] tokens) => tokens.Any(token => parts.Contains(token, StringComparer.OrdinalIgnoreCase));
        bool ContainsName(string name) => stem.Contains(name, StringComparison.OrdinalIgnoreCase);

        if (ContainsName("thai") || HasToken("th"))
            return ["th", "en"];
        if (ContainsName("japanese") || HasToken("ja"))
            return ["ja", "en"];
        if (ContainsName("korean") || HasToken("ko"))
            return ["ko", "en"];
        if (ContainsName("telugu") || HasToken("te"))
            return ["te", "en"];
        if (ContainsName("kannada") || HasToken("kn"))
            return ["kn", "en"];
        if (ContainsName("tamil") || HasToken("ta"))
            return ["ta", "en"];
        if (ContainsName("bengali") || HasToken("bn"))
            return ["bn", "as", "en"];
        if (ContainsName("arabic") || HasToken("ar"))
            return ["ar", "fa", "ur", "ug", "en"];
        if (ContainsName("devanagari") || HasToken("hi", "mr", "ne"))
            return ["hi", "mr", "ne", "en"];
        if (ContainsName("cyrillic") || HasToken("ru"))
            return ["ru", "rs_cyrillic", "be", "bg", "uk", "mn", "en"];
        if (ContainsName("chinese_sim") || ContainsName("zh_sim") || ContainsName("ch_sim"))
            return ["ch_sim", "en"];
        if (ContainsName("chinese_tra") || ContainsName("zh_tra") || ContainsName("ch_tra"))
            return ["ch_tra", "en"];
        if (ContainsName("english") || ContainsName("latin"))
            return ["en"];
        return ["en"];
    }

    static Dictionary<string, (char Start, char End)> GetDefaultSeparatorCharMap()
    {
        return new Dictionary<string, (char Start, char End)>(StringComparer.OrdinalIgnoreCase)
        {
            ["th"] = ('\u00A2', '\u00A3'),
            ["en"] = ('\u00A4', '\u00A5')
        };
    }

    static void RecognizeTextBatch(
        InferenceSession session,
        string inputName,
        byte[] rgb,
        int srcW,
        int srcH,
        List<OcrBox> boxes,
        int targetHeight,
        string charset,
        bool enableWordBeamSearch,
        bool allowRotatedRecognition,
        int beamWidth,
        DecoderLexicon decoderLexicon)
    {
        var baseSamples = new List<RecognitionSample>();
        var zeroAngleConfidences = Enumerable.Repeat(-1f, boxes.Count).ToArray();
        try
        {
            for (int i = 0; i < boxes.Count; i++)
            {
                Vector2[] quad = EnsureQuad(boxes[i]);
                bool isFreeForm = IsFreeFormQuad(quad);
                var (gray, cropW, cropH) = BuildRecognitionCrop(rgb, srcW, srcH, boxes[i]);
                if (cropW <= 0 || cropH <= 0 || gray.IsEmpty)
                {
                    ReturnGrayBuffer(gray);
                    continue;
                }

                var sample = new RecognitionSample
                {
                    BoxIndex = i,
                    Gray = gray,
                    Width = cropW,
                    Height = cropH,
                    IsFreeForm = isFreeForm,
                    OriginalWidth = Math.Max(1, boxes[i].Width),
                    OriginalHeight = Math.Max(1, boxes[i].Height),
                    OriginalIsFreeForm = isFreeForm
                };
                baseSamples.Add(sample);
                boxes[i].DebugRotationPolicy = GetRotationPolicy(sample, allowRotatedRecognition);
            }

            int[] rotationAngles = allowRotatedRecognition ? RecogRotationAngles : [0];
            foreach (int angle in rotationAngles)
            {
                var rotatedSamples = new List<RecognitionSample>(baseSamples.Count);
                try
                {
                    foreach (var sample in baseSamples)
                    {
                        if (ShouldSkipRotationAngle(sample, angle))
                            continue;

                        GrayBuffer rotated = RotateGray(sample.Gray.Buffer, sample.Gray.Length, sample.Width, sample.Height, angle, out int rotatedW, out int rotatedH);
                        rotatedSamples.Add(new RecognitionSample
                        {
                            BoxIndex = sample.BoxIndex,
                            Gray = rotated,
                            Width = rotatedW,
                            Height = rotatedH,
                            IsFreeForm = sample.IsFreeForm,
                            OriginalWidth = sample.OriginalWidth,
                            OriginalHeight = sample.OriginalHeight,
                            OriginalIsFreeForm = sample.OriginalIsFreeForm
                        });
                    }

                    if (rotatedSamples.Count == 0)
                        continue;

                    rotatedSamples.Sort((left, right) =>
                    {
                        int widthCompare = GetRecognitionResizedWidth(left, targetHeight)
                            .CompareTo(GetRecognitionResizedWidth(right, targetHeight));
                        return widthCompare != 0
                            ? widthCompare
                            : left.BoxIndex.CompareTo(right.BoxIndex);
                    });

                    for (int chunkStart = 0; chunkStart < rotatedSamples.Count; chunkStart += RecogBatchSize)
                    {
                        int chunkCount = Math.Min(RecogBatchSize, rotatedSamples.Count - chunkStart);
                        var batch = new List<RecognitionSample>(chunkCount);
                        for (int i = 0; i < chunkCount; i++)
                            batch.Add(rotatedSamples[chunkStart + i]);

                        var predictions = RecognizeTextBatchPass(session, inputName, batch, targetHeight, charset, enableWordBeamSearch, beamWidth, decoderLexicon);
                        var lowConfidence = new List<RecognitionSample>();
                        var lowConfidencePositions = new List<int>();
                        try
                        {
                            for (int i = 0; i < predictions.Count; i++)
                            {
                                if (predictions[i].conf >= RecogContrastThreshold)
                                    continue;

                                GrayBuffer adjusted = AdjustContrastGray(batch[i].Gray.Buffer, batch[i].Gray.Length, batch[i].Width, batch[i].Height, target: RecogAdjustContrast);
                                lowConfidence.Add(new RecognitionSample
                                {
                                    BoxIndex = batch[i].BoxIndex,
                                    Gray = adjusted,
                                    Width = batch[i].Width,
                                    Height = batch[i].Height,
                                    IsFreeForm = batch[i].IsFreeForm,
                                    OriginalWidth = batch[i].OriginalWidth,
                                    OriginalHeight = batch[i].OriginalHeight,
                                    OriginalIsFreeForm = batch[i].OriginalIsFreeForm
                                });
                                lowConfidencePositions.Add(i);
                            }

                            if (lowConfidence.Count > 0)
                            {
                                var retryPredictions = RecognizeTextBatchPass(session, inputName, lowConfidence, targetHeight, charset, enableWordBeamSearch, beamWidth, decoderLexicon);
                                for (int i = 0; i < retryPredictions.Count; i++)
                                {
                                    int position = lowConfidencePositions[i];
                                    if (retryPredictions[i].conf > predictions[position].conf)
                                        predictions[position] = retryPredictions[i];
                                }
                            }
                        }
                        finally
                        {
                            ReleaseRecognitionSamples(lowConfidence);
                        }

                        for (int i = 0; i < predictions.Count; i++)
                        {
                            var prediction = predictions[i];
                            var sample = batch[i];
                            var box = boxes[prediction.boxIndex];
                            if (angle == 0)
                                zeroAngleConfidences[prediction.boxIndex] = Math.Max(zeroAngleConfidences[prediction.boxIndex], prediction.conf);

                            if (ShouldVetoNonZeroAngle(sample, angle, prediction.conf, zeroAngleConfidences[prediction.boxIndex]))
                            {
                                DebugLog($"[SeasonOCR] non-zero-angle veto box#{prediction.boxIndex:D3} angle={angle} conf={prediction.conf:F4} baseline0={zeroAngleConfidences[prediction.boxIndex]:F4}");
                                continue;
                            }

                            if (prediction.conf > box.Confidence)
                            {
                                box.Text = prediction.text;
                                box.Confidence = prediction.conf;
                                box.DebugWinningAngle = angle;
                            }
                        }
                    }
                }
                finally
                {
                    ReleaseRecognitionSamples(rotatedSamples);
                }
            }

            foreach (var sample in baseSamples)
            {
                var box = boxes[sample.BoxIndex];
                DebugLog($"[SeasonOCR] angle winner box#{sample.BoxIndex:D3} angle={box.DebugWinningAngle} policy={box.DebugRotationPolicy} conf={box.Confidence:F4} text=\"{box.Text}\"");
            }
        }
        finally
        {
            ReleaseRecognitionSamples(baseSamples);
        }
    }

    static void ReleaseRecognitionSamples(List<RecognitionSample> samples)
    {
        foreach (var sample in samples)
            ReturnGrayBuffer(sample.Gray);
    }

    static (GrayBuffer gray, int width, int height) BuildRecognitionCrop(
        byte[] rgb, int srcW, int srcH, OcrBox box)
    {
        Vector2[] quad = EnsureQuad(box);
        if (IsFreeFormQuad(quad))
        {
            Vector2[] expandedQuad = ExpandFreeFormQuad(quad, srcW, srcH);
            if (ShouldUseBoundingRectCropForFreeForm(expandedQuad))
                return CropQuadBoundsToGray(rgb, srcW, srcH, expandedQuad);

            return WarpPerspectiveToGray(rgb, srcW, srcH, expandedQuad);
        }

        int marginX = Math.Max(1, box.Width / 20);
        int marginY = Math.Max(1, box.Height / 10);
        int cx = Math.Max(0, box.X - marginX);
        int cy = Math.Max(0, box.Y - marginY);
        int cw = Math.Min(srcW - cx, box.Width + marginX * 2);
        int ch = Math.Min(srcH - cy, box.Height + marginY * 2);
        if (cw <= 0 || ch <= 0)
            return (GrayBuffer.Empty, 0, 0);

        return (RgbToGray(rgb, srcW, srcH, cx, cy, cw, ch), cw, ch);
    }

    static bool IsFreeFormQuad(Vector2[] quad)
    {
        if (quad.Length != 4)
            return false;

        float slopeUp = (quad[1].Y - quad[0].Y) / Math.Max(10f, quad[1].X - quad[0].X);
        float slopeDown = (quad[2].Y - quad[3].Y) / Math.Max(10f, quad[2].X - quad[3].X);
        return Math.Max(Math.Abs(slopeUp), Math.Abs(slopeDown)) >= GroupSlopeThreshold;
    }

    static Vector2[] ExpandFreeFormQuad(Vector2[] quad, int srcW, int srcH)
    {
        quad = OrderQuad(quad);
        Vector2 center = new(
            quad.Average(p => p.X),
            quad.Average(p => p.Y));

        float widthA = Vector2.Distance(quad[2], quad[3]);
        float widthB = Vector2.Distance(quad[1], quad[0]);
        float heightA = Vector2.Distance(quad[1], quad[2]);
        float heightB = Vector2.Distance(quad[0], quad[3]);
        float meanWidth = Math.Max(1f, 0.5f * (widthA + widthB));
        float meanHeight = Math.Max(1f, 0.5f * (heightA + heightB));
        float minSide = Math.Min(meanWidth, meanHeight);
        float scale = minSide <= 80f ? 1.18f : 1.10f;

        var expanded = new Vector2[quad.Length];
        for (int i = 0; i < quad.Length; i++)
        {
            Vector2 offset = quad[i] - center;
            Vector2 candidate = center + offset * scale;
            expanded[i] = new Vector2(
                Math.Clamp(candidate.X, 0f, Math.Max(0, srcW - 1)),
                Math.Clamp(candidate.Y, 0f, Math.Max(0, srcH - 1)));
        }

        return expanded;
    }

    static bool ShouldUseBoundingRectCropForFreeForm(Vector2[] quad)
    {
        quad = OrderQuad(quad);
        float widthA = Vector2.Distance(quad[2], quad[3]);
        float widthB = Vector2.Distance(quad[1], quad[0]);
        float heightA = Vector2.Distance(quad[1], quad[2]);
        float heightB = Vector2.Distance(quad[0], quad[3]);
        float meanWidth = Math.Max(1f, 0.5f * (widthA + widthB));
        float meanHeight = Math.Max(1f, 0.5f * (heightA + heightB));
        float minSide = Math.Min(meanWidth, meanHeight);
        float maxSide = Math.Max(meanWidth, meanHeight);
        float aspect = maxSide / minSide;
        return minSide <= 80f && aspect <= 1.35f;
    }

    static (GrayBuffer gray, int width, int height) CropQuadBoundsToGray(
        byte[] rgb, int srcW, int srcH, Vector2[] quad)
    {
        float minXf = quad.Min(p => p.X);
        float maxXf = quad.Max(p => p.X);
        float minYf = quad.Min(p => p.Y);
        float maxYf = quad.Max(p => p.Y);

        int baseWidth = Math.Max(1, (int)MathF.Ceiling(maxXf - minXf));
        int baseHeight = Math.Max(1, (int)MathF.Ceiling(maxYf - minYf));
        int marginX = Math.Max(2, baseWidth / 12);
        int marginY = Math.Max(2, baseHeight / 12);

        int x0 = Math.Max(0, (int)MathF.Floor(minXf) - marginX);
        int y0 = Math.Max(0, (int)MathF.Floor(minYf) - marginY);
        int x1 = Math.Min(srcW, (int)MathF.Ceiling(maxXf) + marginX);
        int y1 = Math.Min(srcH, (int)MathF.Ceiling(maxYf) + marginY);
        int width = x1 - x0;
        int height = y1 - y0;
        if (width <= 0 || height <= 0)
            return (GrayBuffer.Empty, 0, 0);

        return (RgbToGray(rgb, srcW, srcH, x0, y0, width, height), width, height);
    }

    static (GrayBuffer gray, int width, int height) WarpPerspectiveToGray(
        byte[] rgb, int srcW, int srcH, Vector2[] quad)
    {
        quad = OrderQuad(quad);
        float widthA = Vector2.Distance(quad[2], quad[3]);
        float widthB = Vector2.Distance(quad[1], quad[0]);
        int dstW = Math.Max(1, (int)MathF.Round(Math.Max(widthA, widthB)));
        float heightA = Vector2.Distance(quad[1], quad[2]);
        float heightB = Vector2.Distance(quad[0], quad[3]);
        int dstH = Math.Max(1, (int)MathF.Round(Math.Max(heightA, heightB)));
        if (dstW <= 1 || dstH <= 1)
            return (GrayBuffer.Empty, 0, 0);

        Vector2[] dstRect =
        [
            new Vector2(0, 0),
            new Vector2(dstW - 1, 0),
            new Vector2(dstW - 1, dstH - 1),
            new Vector2(0, dstH - 1)
        ];

        double[] homography = SolvePerspectiveTransform(dstRect, quad);
        GrayBuffer result = RentGrayBuffer(dstW * dstH);
        for (int y = 0; y < dstH; y++)
        {
            for (int x = 0; x < dstW; x++)
            {
                Vector2 src = ApplyPerspectiveTransform(homography, x, y);
                result.Buffer[y * dstW + x] = SampleGrayBilinear(rgb, srcW, srcH, src.X, src.Y);
            }
        }
        return (result, dstW, dstH);
    }

    static bool ShouldSkipRotationAngle(RecognitionSample sample, int angle)
    {
        if (angle == 0)
            return false;

        if (ShouldForceZeroDegreeOnly(sample))
            return true;

        if (angle != 90 && angle != 270)
            return false;

        float minSide = Math.Min(sample.OriginalWidth, sample.OriginalHeight);
        float maxSide = Math.Max(sample.OriginalWidth, sample.OriginalHeight);
        if (minSide <= 0f)
            return false;

        float aspect = maxSide / minSide;
        return minSide <= 80f && aspect <= 1.35f;
    }

    static bool ShouldForceZeroDegreeOnly(RecognitionSample sample)
    {
        if (!sample.OriginalIsFreeForm)
            return false;

        float minSide = Math.Min(sample.OriginalWidth, sample.OriginalHeight);
        float maxSide = Math.Max(sample.OriginalWidth, sample.OriginalHeight);
        if (minSide <= 0f)
            return false;

        float aspect = maxSide / minSide;
        return minSide <= 80f && aspect <= 1.35f;
    }

    static string GetRotationPolicy(RecognitionSample sample, bool allowRotatedRecognition)
    {
        if (!allowRotatedRecognition)
            return "off";

        if (ShouldForceZeroDegreeOnly(sample))
            return "0-only-freeform";

        float minSide = Math.Min(sample.OriginalWidth, sample.OriginalHeight);
        float maxSide = Math.Max(sample.OriginalWidth, sample.OriginalHeight);
        if (minSide > 0f && minSide <= 80f && maxSide / minSide <= 1.35f)
            return "0-180";

        return "0-90-180-270";
    }

    static bool ShouldVetoNonZeroAngle(RecognitionSample sample, int angle, float confidence, float zeroAngleConfidence)
    {
        if (angle == 0 || zeroAngleConfidence < 0f)
            return false;

        if (confidence >= 0.15f)
            return false;

        // Preserve the baseline when a rotated candidate only wins by a tiny amount at very low confidence.
        if (confidence <= zeroAngleConfidence + 0.05f)
            return true;

        return sample.OriginalWidth <= 80 && sample.OriginalHeight <= 80;
    }

    static GrayBuffer RotateGray(byte[] src, int srcLength, int srcW, int srcH, int angle, out int dstW, out int dstH)
    {
        angle = ((angle % 360) + 360) % 360;
        if (angle == 0)
        {
            dstW = srcW;
            dstH = srcH;
            return new GrayBuffer(src, srcLength, pooled: false);
        }

        if (angle == 180)
        {
            dstW = srcW;
            dstH = srcH;
            GrayBuffer rotated = RentGrayBuffer(srcLength);
            for (int y = 0; y < srcH; y++)
            {
                for (int x = 0; x < srcW; x++)
                {
                    int srcIdx = y * srcW + x;
                    int dstIdx = (dstH - 1 - y) * dstW + (dstW - 1 - x);
                    rotated.Buffer[dstIdx] = src[srcIdx];
                }
            }
            return rotated;
        }

        dstW = srcH;
        dstH = srcW;
        GrayBuffer result = RentGrayBuffer(dstW * dstH);
        for (int y = 0; y < srcH; y++)
        {
            for (int x = 0; x < srcW; x++)
            {
                int srcIdx = y * srcW + x;
                int dx, dy;
                if (angle == 90)
                {
                    dx = srcH - 1 - y;
                    dy = x;
                }
                else
                {
                    dx = y;
                    dy = srcW - 1 - x;
                }
                result.Buffer[dy * dstW + dx] = src[srcIdx];
            }
        }
        return result;
    }

    static List<(int boxIndex, string text, float conf)> RecognizeTextBatchPass(
        InferenceSession session,
        string inputName,
        List<RecognitionSample> samples,
        int targetHeight,
        string charset,
        bool enableWordBeamSearch,
        int beamWidth,
        DecoderLexicon decoderLexicon)
    {
        if (samples.Count == 0)
            return [];

        var resizedWidths = new int[samples.Count];
        int maxWidth = RecogMinWidth;
        for (int i = 0; i < samples.Count; i++)
        {
            int resizedW = GetRecognitionResizedWidth(samples[i], targetHeight);
            resizedWidths[i] = resizedW;
            maxWidth = Math.Max(maxWidth, resizedW);
        }

        int inputLength = samples.Count * targetHeight * maxWidth;
        float[] inputData = RentFloatBuffer(inputLength);
        float[] resizedBuffer = RentFloatBuffer(targetHeight * maxWidth);
        try
        {
            for (int i = 0; i < samples.Count; i++)
            {
                ResizeGray(samples[i].Gray.Buffer, samples[i].Width, samples[i].Height, resizedWidths[i], targetHeight, resizedBuffer);
                int batchOffset = i * targetHeight * maxWidth;
                for (int y = 0; y < targetHeight; y++)
                {
                    int dstRow = batchOffset + y * maxWidth;
                    int srcRow = y * resizedWidths[i];
                    float padValue = 0f;
                    for (int x = 0; x < resizedWidths[i]; x++)
                    {
                        padValue = resizedBuffer[srcRow + x] / 127.5f - 1.0f;
                        inputData[dstRow + x] = padValue;
                    }
                    for (int x = resizedWidths[i]; x < maxWidth; x++)
                        inputData[dstRow + x] = padValue;
                }
            }

            var input = new DenseTensor<float>(
                new Memory<float>(inputData, 0, inputLength),
                new[] { samples.Count, 1, targetHeight, maxWidth });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, input)
            };

            using var results = session.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();
            var outDims = outputTensor.Dimensions.ToArray();
            GetRecognizerLayout(outDims, samples.Count, out int batchSize, out int numT, out int numClasses, out RecognizerOutputLayout layout);

            DebugLog($"[SeasonOCR] recognizer output dims=[{string.Join(",", outDims)}] -> N={batchSize} T={numT} C={numClasses}");

            int blankIdx = DetectBlankIndex(outputTensor, 0, numT, numClasses, outDims, layout);
            int expectedChars = numClasses - 1;
            charset = NormalizeCharsetLength(charset, expectedChars);

            var ignoreMask = BuildChineseIgnoreMask(charset, blankIdx, numClasses);
            var predictions = new List<(int boxIndex, string text, float conf)>(samples.Count);
            for (int n = 0; n < samples.Count; n++)
            {
                var probs = BuildRecognizerProbabilityMatrix(outputTensor, outDims, layout, n, numT, numClasses, blankIdx, ignoreMask);
                bool useWordBeamSearch = ShouldUseWordBeamSearch(samples[n], targetHeight, enableWordBeamSearch, decoderLexicon);
                var (decoded, maxProbs) = DecodeRecognition(probs, charset, blankIdx, useWordBeamSearch, beamWidth, ignoreMask, decoderLexicon);
                float confidence = ComputeCustomMeanConfidence(maxProbs);
                predictions.Add((samples[n].BoxIndex, decoded, confidence));
            }

            return predictions;
        }
        finally
        {
            ReturnFloatBuffer(resizedBuffer);
            ReturnFloatBuffer(inputData);
        }
    }

    static void GetRecognizerLayout(
        int[] outDims,
        int requestedBatch,
        out int batchSize,
        out int numT,
        out int numClasses,
        out RecognizerOutputLayout layout)
    {
        if (outDims.Length == 3)
        {
            if (outDims[0] == requestedBatch)
            {
                batchSize = outDims[0];
                numT = outDims[1];
                numClasses = outDims[2];
                layout = RecognizerOutputLayout.Ntc;
                return;
            }

            if (outDims[1] == requestedBatch)
            {
                batchSize = outDims[1];
                numT = outDims[0];
                numClasses = outDims[2];
                layout = RecognizerOutputLayout.Tnc;
                return;
            }

            if (outDims[0] == 1)
            {
                batchSize = 1;
                numT = outDims[1];
                numClasses = outDims[2];
                layout = RecognizerOutputLayout.Ntc;
                return;
            }

            if (outDims[1] == 1)
            {
                batchSize = 1;
                numT = outDims[0];
                numClasses = outDims[2];
                layout = RecognizerOutputLayout.Tnc;
                return;
            }

            if (outDims[0] > 1)
            {
                batchSize = outDims[0];
                numT = outDims[1];
                numClasses = outDims[2];
                layout = RecognizerOutputLayout.Ntc;
                return;
            }
        }

        if (outDims.Length >= 2)
        {
            batchSize = 1;
            numT = outDims[^2];
            numClasses = outDims[^1];
            layout = RecognizerOutputLayout.Tc;
            return;
        }

        batchSize = 1;
        numT = outDims.Length > 0 ? outDims[0] : 0;
        numClasses = Charset.Length + 1;
        layout = RecognizerOutputLayout.Tc;
    }

    static int DetectBlankIndex(
        Tensor<float> outputTensor,
        int batchIndex,
        int numT,
        int numClasses,
        int[] outDims,
        RecognizerOutputLayout layout)
    {
        int cnt0 = 0, cntLast = 0;
        int sampleFrames = Math.Min(8, numT);
        for (int t = 0; t < sampleFrames; t++)
        {
            int best = 0;
            float bestLogit = float.MinValue;
            for (int c = 0; c < numClasses; c++)
            {
                float logit = GetRecognizerLogit(outputTensor, outDims, layout, batchIndex, t, c);
                if (logit > bestLogit)
                {
                    bestLogit = logit;
                    best = c;
                }
            }

            if (best == 0) cnt0++;
            else if (best == numClasses - 1) cntLast++;
        }

        int blankIdx = cnt0 >= cntLast ? 0 : (numClasses - 1);
        DebugLog($"[SeasonOCR]   auto blankIdx={blankIdx} (front frames: 0x{cnt0} last x{cntLast})");
        return blankIdx;
    }

    static string NormalizeCharsetLength(string charset, int expectedChars)
    {
        charset ??= Charset;
        if (charset.Length == expectedChars)
            return charset;

        if (charset.Length < expectedChars)
        {
            var sb = new StringBuilder(charset);
            while (sb.Length < expectedChars)
                sb.Append('?');
            DebugLog($"[SeasonOCR]   charset padded: {charset.Length} -> {expectedChars}");
            return sb.ToString();
        }

        DebugLog($"[SeasonOCR]   charset truncated: {charset.Length} -> {expectedChars}");
        return charset.Substring(0, expectedChars);
    }

    /// <summary>
    /// Converts RGB data to single-channel grayscale.
    /// </summary>
    static GrayBuffer RgbToGray(byte[] rgb, int srcW, int srcH,
        int cropX, int cropY, int cropW, int cropH)
    {
        GrayBuffer gray = RentGrayBuffer(cropW * cropH);
        for (int y = 0; y < cropH; y++)
        {
            for (int x = 0; x < cropW; x++)
            {
                int sx = cropX + x;
                int sy = cropY + y;
                int srcIdx = (sy * srcW + sx) * 3;
                // Weighted grayscale: 0.299R + 0.587G + 0.114B.
                int val = (rgb[srcIdx] * 77 + rgb[srcIdx + 1] * 150 + rgb[srcIdx + 2] * 29) >> 8;
                gray.Buffer[y * cropW + x] = (byte)Math.Clamp(val, 0, 255);
            }
        }
        return gray;
    }

    /// <summary>
    /// Resizes a grayscale image with Catmull-Rom bicubic interpolation.
    /// Quality is intended to match EasyOCR's bicubic or lanczos-style preprocessing.
    /// </summary>
    static void ResizeGray(byte[] src, int srcW, int srcH, int dstW, int dstH, float[] destination)
    {
        float scaleX = (float)srcW / dstW;
        float scaleY = (float)srcH / dstH;

        for (int dy = 0; dy < dstH; dy++)
        {
            float sy = dy * scaleY;
            int y0 = (int)sy;

            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = dx * scaleX;
                int x0 = (int)sx;

                float sum = 0;
                float weightSum = 0;
                for (int ky = -1; ky <= 2; ky++)
                {
                    int syIdx = y0 + ky;
                    if ((uint)syIdx >= srcH) continue;
                    float wy = CatmullRom(ky - (sy - y0));

                    for (int kx = -1; kx <= 2; kx++)
                    {
                        int sxIdx = x0 + kx;
                        if ((uint)sxIdx >= srcW) continue;
                        float w = wy * CatmullRom(kx - (sx - x0));
                        sum += src[syIdx * srcW + sxIdx] * w;
                        weightSum += w;
                    }
                }
                destination[dy * dstW + dx] = weightSum > 0 ? sum / weightSum : src[y0 * srcW + x0];
            }
        }
    }

    /// <summary>
    /// Catmull-Rom bicubic kernel.
    /// </summary>
    static float CatmullRom(float x)
    {
        float ax = Math.Abs(x);
        if (ax <= 1f) return (1.5f * ax - 2.5f) * ax * ax + 1f;
        if (ax < 2f) return ((-0.5f * ax + 2.5f) * ax - 4f) * ax + 2f;
        return 0f;
    }

    /// <summary>
    /// Applies contrast stretching similar to EasyOCR's adjust_contrast_grey.
    /// When the image contrast falls below the target value, the histogram is stretched
    /// to improve text readability.
    /// </summary>
    static GrayBuffer AdjustContrastGray(byte[] gray, int grayLength, int w, int h, float target = 0.4f)
    {
        // Estimate percentile bounds.
        var hist = new int[256];
        for (int i = 0; i < grayLength; i++)
            hist[gray[i]]++;

        int total = grayLength;
        int p10 = 0, p90 = 255;
        int cum = 0;
        int thresh10 = (int)(total * 0.10f);
        int thresh90 = (int)(total * 0.90f);
        bool found10 = false;
        for (int v = 0; v < 256; v++)
        {
            cum += hist[v];
            if (!found10 && cum >= thresh10) { p10 = v; found10 = true; }
            if (cum >= thresh90) { p90 = v; break; }
        }

        float high = p90, low = p10;
        float contrast = (high - low) / Math.Max(10, high + low);

        if (contrast < target)
        {
            GrayBuffer result = RentGrayBuffer(grayLength);
            float ratio = 200f / Math.Max(10, high - low);
            for (int i = 0; i < grayLength; i++)
            {
                int v = (int)((gray[i] - low + 25) * ratio);
                result.Buffer[i] = (byte)Math.Clamp(v, 0, 255);
            }
            return result;
        }
        return new GrayBuffer(gray, grayLength, pooled: false);
    }

    static float[,] BuildRecognizerProbabilityMatrix(
        Tensor<float> output,
        int[] outDims,
        RecognizerOutputLayout layout,
        int batchIndex,
        int tCount,
        int classCount,
        int blankIdx,
        bool[]? ignoreMask)
    {
        var probs = new float[tCount, classCount];
        var stepProb = new float[classCount];

        for (int t = 0; t < tCount; t++)
        {
            float maxLogit = float.MinValue;
            for (int c = 0; c < classCount; c++)
            {
                float logit = GetRecognizerLogit(output, outDims, layout, batchIndex, t, c);
                stepProb[c] = logit;
                if (ignoreMask != null && c < ignoreMask.Length && ignoreMask[c])
                    continue;
                if (logit > maxLogit)
                    maxLogit = logit;
            }

            float sum = 0f;
            for (int c = 0; c < classCount; c++)
            {
                if (ignoreMask != null && c < ignoreMask.Length && ignoreMask[c])
                {
                    probs[t, c] = 0f;
                    continue;
                }

                float prob = MathF.Exp(stepProb[c] - maxLogit);
                probs[t, c] = prob;
                sum += prob;
            }

            if (sum <= 0f)
            {
                probs[t, blankIdx] = 1f;
                continue;
            }

            for (int c = 0; c < classCount; c++)
                probs[t, c] /= sum;
        }

        return probs;
    }

    static (string text, List<float> maxProbs) DecodeRecognition(
        float[,] probs,
        string charset,
        int blankIdx,
        bool useWordBeamSearch,
        int beamWidth,
        bool[]? ignoreMask,
        DecoderLexicon decoderLexicon)
    {
        var (beamProbs, classes, ignoreIndices) = RemapForBeamSearch(probs, charset, blankIdx, ignoreMask, decoderLexicon);
        List<float> maxProbs = CollectFrameMaxProbs(beamProbs, blankIndex: 0);
        string decoded = useWordBeamSearch
            ? DecodeWordBeamSearch(beamProbs, classes, ignoreIndices, beamWidth, decoderLexicon)
            : CtcBeamSearch(beamProbs, classes, ignoreIndices, beamWidth);

        return (decoded.Trim(), maxProbs);
    }

    static bool ShouldUseWordBeamSearch(
        RecognitionSample sample,
        int targetHeight,
        bool enableWordBeamSearch,
        DecoderLexicon decoderLexicon)
    {
        if (!enableWordBeamSearch || decoderLexicon == null || decoderLexicon.CombinedWords.Count == 0)
            return false;

        float width = Math.Max(1, sample.Width);
        float height = Math.Max(1, sample.Height);
        float minSide = Math.Min(width, height);
        float maxSide = Math.Max(width, height);
        float aspect = maxSide / minSide;
        int resizedWidth = Math.Max(RecogMinWidth, (int)Math.Ceiling(targetHeight * (width / height)));

        // Very small or near-square sign crops are prone to false dictionary-biased matches like "E" -> "UU".
        if (minSide <= 80f && aspect <= 1.35f)
            return false;

        if (resizedWidth <= 96)
            return false;

        return true;
    }

    static string DecodeWordBeamSearch(
        float[,] probs,
        string[] classes,
        HashSet<int> ignoreIndices,
        int beamWidth,
        DecoderLexicon decoderLexicon)
    {
        if (decoderLexicon == null || decoderLexicon.CombinedWords.Count == 0)
            return CtcBeamSearch(probs, classes, ignoreIndices, beamWidth);

        var argmax = BuildArgmaxSequence(probs);
        if (decoderLexicon.HasSeparators)
        {
            var segments = BuildSeparatorSegments(argmax, classes, decoderLexicon);
            if (segments.Count == 0)
                return CtcBeamSearch(probs, classes, ignoreIndices, beamWidth, decoderLexicon.CombinedWords, "wordbeam/fallback-no-segments");

            var text = new StringBuilder();
            for (int i = 0; i < segments.Count; i++)
            {
                var segment = segments[i];
                float[,] slice = SliceProbabilityMatrix(probs, segment.StartIndex, segment.EndIndex + 1);
                HashSet<string>? dictionary = GetSegmentDictionary(decoderLexicon, segment.Language);
                string debugLabel = $"wordbeam/segment#{i + 1} lang={(string.IsNullOrWhiteSpace(segment.Language) ? "default" : segment.Language)} range={segment.StartIndex}-{segment.EndIndex}";
                text.Append(DecodeBeamSlice(slice, classes, ignoreIndices, beamWidth, dictionary, debugLabel));
            }
            return text.ToString();
        }

        int spaceIdx = Array.FindIndex(classes, c => c == " ");
        if (spaceIdx < 0)
            return CtcBeamSearch(probs, classes, ignoreIndices, beamWidth, decoderLexicon.CombinedWords, "wordbeam/no-space");

        int tCount = probs.GetLength(0);
        var nonSpaceIndices = new List<int>();
        for (int t = 0; t < tCount; t++)
        {
            if (argmax[t] != spaceIdx)
                nonSpaceIndices.Add(t);
        }

        if (nonSpaceIndices.Count == 0)
            return string.Empty;

        var groups = SplitConsecutiveIndices(nonSpaceIndices);
        var words = new List<string>();
        for (int i = 0; i < groups.Count; i++)
        {
            var group = groups[i];
            float[,] slice = SliceProbabilityMatrix(probs, group[0], group[^1] + 1);
            HashSet<string>? dictionary = ResolveDictionaryForRange(argmax, group[0], group[^1], classes, decoderLexicon);
            words.Add(DecodeBeamSlice(slice, classes, ignoreIndices, beamWidth, dictionary, $"wordbeam/group#{i + 1} range={group[0]}-{group[^1]}"));
        }

        return string.Join(" ", words.Where(w => !string.IsNullOrWhiteSpace(w)));
    }

    static int[] BuildArgmaxSequence(float[,] probs)
    {
        int tCount = probs.GetLength(0);
        int classCount = probs.GetLength(1);
        var argmax = new int[tCount];
        for (int t = 0; t < tCount; t++)
        {
            int bestIdx = 0;
            float bestProb = probs[t, 0];
            for (int c = 1; c < classCount; c++)
            {
                if (probs[t, c] <= bestProb)
                    continue;
                bestProb = probs[t, c];
                bestIdx = c;
            }
            argmax[t] = bestIdx;
        }
        return argmax;
    }

    static List<List<int>> SplitConsecutiveIndices(List<int> indices)
    {
        var result = new List<List<int>>();
        if (indices.Count == 0)
            return result;

        var current = new List<int> { indices[0] };
        for (int i = 1; i < indices.Count; i++)
        {
            if (indices[i] == indices[i - 1] + 1)
            {
                current.Add(indices[i]);
            }
            else
            {
                result.Add(current);
                current = new List<int> { indices[i] };
            }
        }
        result.Add(current);
        return result;
    }

    static List<SeparatorSegment> BuildSeparatorSegments(int[] argmax, string[] classes, DecoderLexicon decoderLexicon)
    {
        var separatorPositions = new List<(int Position, string Language, bool IsStart)>();

        foreach (var (lang, separators) in decoderLexicon.SeparatorsByLanguage)
        {
            int startIdx = Array.FindIndex(classes, c => c == separators.Start.ToString());
            int endIdx = Array.FindIndex(classes, c => c == separators.End.ToString());
            if (startIdx < 0 || endIdx < 0)
                continue;

            foreach (int pos in GetSeparatorPositions(argmax, startIdx, useFirst: false))
                separatorPositions.Add((pos, lang, true));
            foreach (int pos in GetSeparatorPositions(argmax, endIdx, useFirst: true))
                separatorPositions.Add((pos, lang, false));
        }

        separatorPositions.Sort((a, b) => a.Position.CompareTo(b.Position));
        var result = new List<SeparatorSegment>();
        int startIdxGlobal = 0;
        string activeLanguage = string.Empty;
        int activeStart = -1;

        foreach (var separator in separatorPositions)
        {
            if (separator.IsStart)
            {
                activeLanguage = separator.Language;
                activeStart = separator.Position;
                continue;
            }

            if (!string.Equals(activeLanguage, separator.Language, StringComparison.OrdinalIgnoreCase))
                continue;

            if (activeStart > startIdxGlobal)
                result.Add(new SeparatorSegment { Language = "", StartIndex = startIdxGlobal, EndIndex = activeStart - 1 });

            result.Add(new SeparatorSegment
            {
                Language = separator.Language,
                StartIndex = activeStart + 1,
                EndIndex = separator.Position - 1
            });

            startIdxGlobal = separator.Position + 1;
            activeLanguage = string.Empty;
            activeStart = -1;
        }

        if (startIdxGlobal <= argmax.Length - 1)
            result.Add(new SeparatorSegment { Language = "", StartIndex = startIdxGlobal, EndIndex = argmax.Length - 1 });

        var filtered = result.Where(s => s.StartIndex <= s.EndIndex).ToList();
        if (filtered.Count == 0)
        {
            DebugLog("[SeasonOCR] separator segments: none");
        }
        else
        {
            DebugLog($"[SeasonOCR] separator segments: {string.Join("; ", filtered.Select(s => $"{(string.IsNullOrWhiteSpace(s.Language) ? "default" : s.Language)}[{s.StartIndex}-{s.EndIndex}]"))}");
        }
        return filtered;
    }

    static List<int> GetSeparatorPositions(int[] argmax, int targetIndex, bool useFirst)
    {
        var matches = new List<int>();
        for (int i = 0; i < argmax.Length; i++)
        {
            if (argmax[i] == targetIndex)
                matches.Add(i);
        }

        if (matches.Count == 0)
            return [];

        var grouped = SplitConsecutiveIndices(matches);
        return grouped.Select(g => useFirst ? g[0] : g[^1]).ToList();
    }

    static string DecodeBeamSlice(
        float[,] probs,
        string[] classes,
        HashSet<int> ignoreIndices,
        int beamWidth,
        HashSet<string>? dictionary,
        string debugLabel)
    {
        if (dictionary == null || dictionary.Count == 0)
        {
            DebugLog($"[SeasonOCR] {debugLabel} fallback=beamsearch");
            return CtcBeamSearch(probs, classes, ignoreIndices, beamWidth);
        }

        return CtcBeamSearch(
            probs,
            classes,
            ignoreIndices,
            beamWidth,
            dictionary,
            $"{debugLabel} dict={dictionary.Count}");
    }

    static HashSet<string>? GetSegmentDictionary(DecoderLexicon decoderLexicon, string language)
    {
        if (!string.IsNullOrWhiteSpace(language) &&
            decoderLexicon.WordsByLanguage.TryGetValue(language, out var words) &&
            words.Count > 0)
        {
            return words;
        }

        return null;
    }

    static HashSet<string>? ResolveDictionaryForRange(
        int[] argmax,
        int startIndex,
        int endIndex,
        string[] classes,
        DecoderLexicon decoderLexicon)
    {
        string? inferredLanguage = InferDictionaryLanguageForRange(argmax, startIndex, endIndex, classes, decoderLexicon);
        return GetSegmentDictionary(decoderLexicon, inferredLanguage ?? string.Empty);
    }

    static string? InferDictionaryLanguageForRange(
        int[] argmax,
        int startIndex,
        int endIndex,
        string[] classes,
        DecoderLexicon decoderLexicon)
    {
        bool hasLatin = false;
        bool hasKana = false;
        bool hasHangul = false;
        bool hasThai = false;
        bool hasTamil = false;
        bool hasTelugu = false;
        bool hasKannada = false;
        bool hasBengali = false;
        bool hasDevanagari = false;
        bool hasArabic = false;
        bool hasCyrillic = false;

        for (int i = startIndex; i <= endIndex && i < argmax.Length; i++)
        {
            int clsIndex = argmax[i];
            if (clsIndex < 0 || clsIndex >= classes.Length)
                continue;

            string value = classes[clsIndex];
            if (string.IsNullOrEmpty(value) || value == "[blank]")
                continue;

            char ch = value[0];
            if ((ch is >= 'A' and <= 'Z') || (ch is >= 'a' and <= 'z'))
                hasLatin = true;
            else if (ch is >= '\u3040' and <= '\u30FF')
                hasKana = true;
            else if ((ch is >= '\uAC00' and <= '\uD7AF') || (ch is >= '\u1100' and <= '\u11FF'))
                hasHangul = true;
            else if (ch is >= '\u0E00' and <= '\u0E7F')
                hasThai = true;
            else if (ch is >= '\u0B80' and <= '\u0BFF')
                hasTamil = true;
            else if (ch is >= '\u0C00' and <= '\u0C7F')
                hasTelugu = true;
            else if (ch is >= '\u0C80' and <= '\u0CFF')
                hasKannada = true;
            else if (ch is >= '\u0980' and <= '\u09FF')
                hasBengali = true;
            else if (ch is >= '\u0900' and <= '\u097F')
                hasDevanagari = true;
            else if ((ch is >= '\u0600' and <= '\u06FF') || (ch is >= '\u0750' and <= '\u077F'))
                hasArabic = true;
            else if ((ch is >= '\u0400' and <= '\u04FF') || (ch is >= '\u0500' and <= '\u052F'))
                hasCyrillic = true;
        }

        if (hasKana)
            return FirstAvailableLanguage(decoderLexicon, "ja");
        if (hasHangul)
            return FirstAvailableLanguage(decoderLexicon, "ko");
        if (hasThai)
            return FirstAvailableLanguage(decoderLexicon, "th");
        if (hasTamil)
            return FirstAvailableLanguage(decoderLexicon, "ta");
        if (hasTelugu)
            return FirstAvailableLanguage(decoderLexicon, "te");
        if (hasKannada)
            return FirstAvailableLanguage(decoderLexicon, "kn");
        if (hasBengali)
            return FirstAvailableLanguage(decoderLexicon, "bn", "as", "mni");
        if (hasDevanagari)
            return FirstAvailableLanguage(decoderLexicon, "hi", "mr", "ne", "bh", "mai", "ang", "bho", "mah", "sck", "new", "gom", "sa", "bgc");
        if (hasArabic)
            return FirstAvailableLanguage(decoderLexicon, "ar", "fa", "ur", "ug");
        if (hasCyrillic)
            return FirstAvailableLanguage(decoderLexicon, "ru", "rs_cyrillic", "be", "bg", "uk", "mn", "abq", "ady", "kbd", "ava", "dar", "inh", "che", "lbe", "lez", "tab", "tjk");
        if (hasLatin)
            return FirstAvailableLanguage(decoderLexicon, "en");

        return null;
    }

    static string? FirstAvailableLanguage(DecoderLexicon decoderLexicon, params string[] candidates)
    {
        foreach (string candidate in candidates)
        {
            if (decoderLexicon.WordsByLanguage.TryGetValue(candidate, out var words) && words.Count > 0)
                return candidate;
        }

        return null;
    }

    static float[,] SliceProbabilityMatrix(float[,] probs, int startInclusive, int endExclusive)
    {
        int classCount = probs.GetLength(1);
        int rows = Math.Max(0, endExclusive - startInclusive);
        var slice = new float[rows, classCount];
        for (int t = 0; t < rows; t++)
        {
            for (int c = 0; c < classCount; c++)
                slice[t, c] = probs[startInclusive + t, c];
        }
        return slice;
    }

    static (float[,] probs, string[] classes, HashSet<int> ignoreIndices) RemapForBeamSearch(
        float[,] probs,
        string charset,
        int blankIdx,
        bool[]? ignoreMask,
        DecoderLexicon decoderLexicon)
    {
        int tCount = probs.GetLength(0);
        int classCount = probs.GetLength(1);
        var remapped = new float[tCount, classCount];
        var classes = new string[classCount];
        classes[0] = "[blank]";
        var ignoreIndices = new HashSet<int> { 0 };

        for (int c = 0; c < classCount; c++)
        {
            int mappedIdx = blankIdx == 0
                ? c
                : (c == blankIdx ? 0 : c + 1);

            for (int t = 0; t < tCount; t++)
                remapped[t, mappedIdx] = probs[t, c];

            if (c == blankIdx)
                continue;

            int charIdx = blankIdx == 0 ? c - 1 : c;
            classes[mappedIdx] = charIdx >= 0 && charIdx < charset.Length ? charset[charIdx].ToString() : "?";
            if (ignoreMask != null && c < ignoreMask.Length && ignoreMask[c])
                ignoreIndices.Add(mappedIdx);
        }

        if (decoderLexicon != null)
        {
            foreach (var separators in decoderLexicon.SeparatorsByLanguage.Values)
            {
                AddCharIndexToIgnore(classes, ignoreIndices, separators.Start);
                AddCharIndexToIgnore(classes, ignoreIndices, separators.End);
            }
        }

        return (remapped, classes, ignoreIndices);
    }

    static void AddCharIndexToIgnore(string[] classes, HashSet<int> ignoreIndices, char value)
    {
        int idx = Array.FindIndex(classes, c => c == value.ToString());
        if (idx >= 0)
            ignoreIndices.Add(idx);
    }

    static List<float> CollectFrameMaxProbs(float[,] probs, int blankIndex)
    {
        int tCount = probs.GetLength(0);
        int classCount = probs.GetLength(1);
        var result = new List<float>();
        for (int t = 0; t < tCount; t++)
        {
            int bestIdx = 0;
            float bestProb = probs[t, 0];
            for (int c = 1; c < classCount; c++)
            {
                if (probs[t, c] <= bestProb)
                    continue;
                bestProb = probs[t, c];
                bestIdx = c;
            }

            if (bestIdx != blankIndex)
                result.Add(bestProb);
        }
        return result;
    }

    static string CtcBeamSearch(
        float[,] probs,
        string[] classes,
        HashSet<int> ignoreIndices,
        int beamWidth,
        HashSet<string>? dictionaryWords = null,
        string? debugContext = null)
    {
        int tCount = probs.GetLength(0);
        int classCount = probs.GetLength(1);
        const int blankIdx = 0;

        var last = new BeamState();
        var empty = new BeamEntry { PrBlank = 1f, PrTotal = 1f, Labeling = [] };
        last.Entries[string.Empty] = empty;

        for (int t = 0; t < tCount; t++)
        {
            var curr = new BeamState();
            var bestLabelings = SortBeamEntries(last).Take(beamWidth).ToList();
            foreach (var prev in bestLabelings)
            {
                int[] labeling = prev.Labeling;
                float prNonBlank = labeling.Length > 0 ? prev.PrNonBlank * probs[t, labeling[^1]] : 0f;
                float prBlank = prev.PrTotal * probs[t, blankIdx];

                var currentEntry = GetOrAddBeam(curr, labeling);
                currentEntry.PrNonBlank += prNonBlank;
                currentEntry.PrBlank += prBlank;
                currentEntry.PrTotal += prBlank + prNonBlank;
                currentEntry.PrText = prev.PrText;

                float threshold = 0.5f / classCount;
                for (int c = 0; c < classCount; c++)
                {
                    if (probs[t, c] < threshold)
                        continue;

                    int[] newLabeling = FastSimplifyLabel(labeling, c, blankIdx);
                    float newPrNonBlank = labeling.Length > 0 && labeling[^1] == c
                        ? probs[t, c] * prev.PrBlank
                        : probs[t, c] * prev.PrTotal;

                    var nextEntry = GetOrAddBeam(curr, newLabeling);
                    nextEntry.PrNonBlank += newPrNonBlank;
                    nextEntry.PrTotal += newPrNonBlank;
                    nextEntry.PrText = prev.PrText;
                }
            }

            last = curr;
        }

        var candidates = SortBeamEntries(last);
        BeamEntry? best = candidates.FirstOrDefault();
        if (best == null)
            return string.Empty;

        if (dictionaryWords != null && dictionaryWords.Count > 0)
        {
            foreach (var candidate in candidates.Take(20))
            {
                string decodedCandidate = DecodeBeamLabeling(candidate.Labeling, classes, ignoreIndices);
                if (dictionaryWords.Contains(decodedCandidate))
                {
                    if (!string.IsNullOrWhiteSpace(debugContext))
                        DebugLog($"[SeasonOCR] {debugContext} dictionary-hit \"{decodedCandidate}\"");
                    return decodedCandidate;
                }
            }

            if (!string.IsNullOrWhiteSpace(debugContext))
                DebugLog($"[SeasonOCR] {debugContext} dictionary-miss, fallback-best \"{DecodeBeamLabeling(best.Labeling, classes, ignoreIndices)}\"");
        }

        return DecodeBeamLabeling(best.Labeling, classes, ignoreIndices);
    }

    static string DecodeBeamLabeling(int[] labeling, string[] classes, HashSet<int> ignoreIndices)
    {
        var text = new StringBuilder();
        for (int i = 0; i < labeling.Length; i++)
        {
            int label = labeling[i];
            if (ignoreIndices.Contains(label))
                continue;
            if (i > 0 && labeling[i - 1] == label)
                continue;
            if (!string.IsNullOrEmpty(classes[label]))
                text.Append(classes[label]);
        }
        return text.ToString();
    }

    static List<BeamEntry> SortBeamEntries(BeamState state)
    {
        return state.Entries.Values
            .OrderByDescending(entry => entry.PrTotal * entry.PrText)
            .ToList();
    }

    static BeamEntry GetOrAddBeam(BeamState state, int[] labeling)
    {
        string key = MakeBeamKey(labeling);
        if (!state.Entries.TryGetValue(key, out var entry))
        {
            entry = new BeamEntry { Labeling = labeling };
            state.Entries[key] = entry;
        }
        return entry;
    }

    static string MakeBeamKey(int[] labeling)
    {
        if (labeling.Length == 0)
            return string.Empty;
        return string.Join(",", labeling);
    }

    static int[] FastSimplifyLabel(int[] labeling, int c, int blankIdx)
    {
        if (labeling.Length > 0 && c == blankIdx && labeling[^1] != blankIdx)
            return AppendValue(labeling, c);

        if (labeling.Length > 0 && c != blankIdx && labeling[^1] == blankIdx)
        {
            if (labeling.Length >= 2 && labeling[^2] == c)
                return AppendValue(labeling, c);
            return ReplaceLastValue(labeling, c);
        }

        if (labeling.Length > 0 && c == blankIdx && labeling[^1] == blankIdx)
            return labeling;

        if (labeling.Length == 0 && c == blankIdx)
            return labeling;

        if (labeling.Length == 0 && c != blankIdx)
            return [c];

        if (labeling.Length > 0 && c != blankIdx)
            return AppendValue(labeling, c);

        return SimplifyLabel(AppendValue(labeling, c), blankIdx);
    }

    static int[] SimplifyLabel(int[] labeling, int blankIdx)
    {
        if (labeling.Length == 0)
            return labeling;

        var collapsed = new List<int>(labeling.Length);
        for (int i = 0; i < labeling.Length; i++)
        {
            bool repeatedBlank = i > 0 && labeling[i - 1] == labeling[i] && labeling[i] == blankIdx;
            if (!repeatedBlank)
                collapsed.Add(labeling[i]);
        }

        var simplified = new List<int>(collapsed.Count);
        for (int i = 0; i < collapsed.Count; i++)
        {
            bool blankBetweenDifferentChars = i > 0 && i < collapsed.Count - 1 &&
                collapsed[i] == blankIdx &&
                collapsed[i - 1] != collapsed[i + 1];
            if (!blankBetweenDifferentChars)
                simplified.Add(collapsed[i]);
        }

        return simplified.ToArray();
    }

    static int[] AppendValue(int[] source, int value)
    {
        var result = new int[source.Length + 1];
        Array.Copy(source, result, source.Length);
        result[^1] = value;
        return result;
    }

    static int[] ReplaceLastValue(int[] source, int value)
    {
        var result = new int[source.Length];
        Array.Copy(source, result, source.Length);
        result[^1] = value;
        return result;
    }

    static float GetRecognizerLogit(
        Tensor<float> output,
        int[] outDims,
        RecognizerOutputLayout layout,
        int batchIndex,
        int t,
        int c)
    {
        if (outDims.Length == 3 && layout == RecognizerOutputLayout.Ntc)
            return output[batchIndex, t, c];
        if (outDims.Length == 3 && layout == RecognizerOutputLayout.Tnc)
            return output[t, batchIndex, c];
        if (outDims.Length == 2)
            return output[t, c];
        if (outDims.Length == 4 && outDims[0] == 1 && outDims[1] == 1)
            return output[0, 0, t, c];
        return output[t, c];
    }

    static float ComputeCustomMeanConfidence(List<float> probs)
    {
        if (probs.Count == 0)
            return 0f;

        double logSum = 0;
        foreach (float prob in probs)
            logSum += Math.Log(Math.Max(1e-6f, prob));
        return (float)Math.Exp(logSum * (2.0 / Math.Sqrt(probs.Count)));
    }

    static double[] SolvePerspectiveTransform(Vector2[] src, Vector2[] dst)
    {
        double[,] a = new double[8, 9];
        for (int i = 0; i < 4; i++)
        {
            double x = src[i].X;
            double y = src[i].Y;
            double u = dst[i].X;
            double v = dst[i].Y;

            int r0 = i * 2;
            int r1 = r0 + 1;
            a[r0, 0] = x; a[r0, 1] = y; a[r0, 2] = 1; a[r0, 6] = -u * x; a[r0, 7] = -u * y; a[r0, 8] = u;
            a[r1, 3] = x; a[r1, 4] = y; a[r1, 5] = 1; a[r1, 6] = -v * x; a[r1, 7] = -v * y; a[r1, 8] = v;
        }

        for (int col = 0; col < 8; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < 8; row++)
            {
                if (Math.Abs(a[row, col]) > Math.Abs(a[pivot, col]))
                    pivot = row;
            }

            if (Math.Abs(a[pivot, col]) < 1e-8)
                return [1, 0, 0, 0, 1, 0, 0, 0, 1];

            if (pivot != col)
            {
                for (int k = col; k < 9; k++)
                    (a[col, k], a[pivot, k]) = (a[pivot, k], a[col, k]);
            }

            double div = a[col, col];
            for (int k = col; k < 9; k++)
                a[col, k] /= div;

            for (int row = 0; row < 8; row++)
            {
                if (row == col)
                    continue;

                double factor = a[row, col];
                if (Math.Abs(factor) < 1e-8)
                    continue;

                for (int k = col; k < 9; k++)
                    a[row, k] -= factor * a[col, k];
            }
        }

        return [a[0, 8], a[1, 8], a[2, 8], a[3, 8], a[4, 8], a[5, 8], a[6, 8], a[7, 8], 1.0];
    }

    static Vector2 ApplyPerspectiveTransform(double[] h, float x, float y)
    {
        double denom = h[6] * x + h[7] * y + h[8];
        if (Math.Abs(denom) < 1e-8)
            return Vector2.Zero;

        float tx = (float)((h[0] * x + h[1] * y + h[2]) / denom);
        float ty = (float)((h[3] * x + h[4] * y + h[5]) / denom);
        return new Vector2(tx, ty);
    }

    static byte SampleGrayBilinear(byte[] rgb, int srcW, int srcH, float x, float y)
    {
        x = Math.Clamp(x, 0, srcW - 1);
        y = Math.Clamp(y, 0, srcH - 1);
        int x0 = (int)MathF.Floor(x);
        int y0 = (int)MathF.Floor(y);
        int x1 = Math.Min(x0 + 1, srcW - 1);
        int y1 = Math.Min(y0 + 1, srcH - 1);
        float fx = x - x0;
        float fy = y - y0;

        float g00 = GrayAt(rgb, srcW, x0, y0);
        float g10 = GrayAt(rgb, srcW, x1, y0);
        float g01 = GrayAt(rgb, srcW, x0, y1);
        float g11 = GrayAt(rgb, srcW, x1, y1);
        float top = g00 + (g10 - g00) * fx;
        float bottom = g01 + (g11 - g01) * fx;
        return (byte)Math.Clamp((int)MathF.Round(top + (bottom - top) * fy), 0, 255);
    }

    static byte GrayAt(byte[] rgb, int srcW, int x, int y)
    {
        int idx = (y * srcW + x) * 3;
        int value = (rgb[idx] * 77 + rgb[idx + 1] * 150 + rgb[idx + 2] * 29) >> 8;
        return (byte)value;
    }

    /// <summary>
    /// Returns an optional class ignore mask for recognizer decoding.
    /// Chinese models are not masked here because their embedded charset often contains
    /// digits, punctuation, and Latin letters that are required for mixed-language OCR.
    /// </summary>
    static bool[]? BuildChineseIgnoreMask(string charset, int blankIdx, int numClasses)
    {
        _ = charset;
        _ = blankIdx;
        _ = numClasses;
        DebugLog("[SeasonOCR]   Chinese ignore mask: disabled");
        return null;
    }
}
