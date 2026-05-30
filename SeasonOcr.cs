// Copyright (c) SeasonRealms and contributors.
// Licensed under the Apache License, Version 2.0.
// SeasonOCR for EasyOCR Models
//
// This file contains a C# translation/adaptation of logic from the EasyOCR project:
// https://github.com/JaidedAI/EasyOCR
// Original project license: Apache-2.0.
// Portions of the detection pipeline trace back to CRAFT-related code distributed under the MIT License.
// Modified from the original Python implementation.

using System.Buffers;

namespace SeasonOCR;

/// <summary>
/// Provides EasyOCR-style text detection and recognition for ONNX models.
/// </summary>
public static partial class SeasonOcr
{
    public static bool EnableDebugOutput { get; set; } = false;
    private static readonly ArrayPool<float> FloatArrayPool = ArrayPool<float>.Shared;
    private static readonly ArrayPool<byte> ByteArrayPool = ArrayPool<byte>.Shared;

    private static void DebugLog(string message)
    {
        if (EnableDebugOutput)
            Debug.WriteLine(message);
    }

    private const string Charset =
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";

    private const int CraftCanvasSize = 2560;
    private const float CraftMagRatio = 1.0f;
    private const float TextThreshHigh = 0.75f;
    private const float TextThreshLow = 0.4f;
    private const float LinkThresh = 0.4f;
    private const int MinComponentSize = 10;
    private const int MinBoxArea = 64;
    private const float GroupYCenterThreshold = 0.5f;
    private const float GroupHeightThreshold = 0.5f;
    private const float GroupWidthThreshold = 1.0f;
    private const float GroupAddMargin = 0.05f;
    private const float GroupSlopeThreshold = 0.1f;
    private const float ParagraphXThreshold = 1.0f;
    private const float ParagraphYThreshold = 0.5f;
    private const float ParagraphLineThreshold = 0.4f;

    private const int RecogHeight = 64;
    private const int RecogMinWidth = 8;
    private const int RecogBatchSize = 16;
    private const float RecogContrastThreshold = 0.1f;
    private const float RecogAdjustContrast = 0.5f;
    private static readonly int[] RecogRotationAngles = [0, 90, 180, 270];

    private enum RecognizerOutputLayout
    {
        Ntc,
        Tnc,
        Tc
    }

    private sealed class BeamEntry
    {
        public float PrTotal { get; set; }
        public float PrNonBlank { get; set; }
        public float PrBlank { get; set; }
        public float PrText { get; set; } = 1f;
        public int[] Labeling { get; set; } = [];
    }

    private sealed class BeamState
    {
        public Dictionary<string, BeamEntry> Entries { get; } = new();
    }

    private sealed class DecoderLexicon
    {
        public HashSet<string> CombinedWords { get; } = new(StringComparer.OrdinalIgnoreCase);
        public Dictionary<string, HashSet<string>> WordsByLanguage { get; } = new(StringComparer.OrdinalIgnoreCase);
        public Dictionary<string, (char Start, char End)> SeparatorsByLanguage { get; } = new(StringComparer.OrdinalIgnoreCase);
        public bool HasSeparators => SeparatorsByLanguage.Count > 0;
    }

    private sealed class SeparatorSegment
    {
        public string Language { get; init; } = "";
        public int StartIndex { get; init; }
        public int EndIndex { get; init; }
    }

    private sealed class RecognitionSample
    {
        public int BoxIndex { get; init; }
        public GrayBuffer Gray { get; init; } = GrayBuffer.Empty;
        public int Width { get; init; }
        public int Height { get; init; }
        public bool IsFreeForm { get; init; }
        public int OriginalWidth { get; init; }
        public int OriginalHeight { get; init; }
        public bool OriginalIsFreeForm { get; init; }
    }

    private readonly struct GrayBuffer(byte[] buffer, int length, bool pooled)
    {
        public static GrayBuffer Empty { get; } = new([], 0, false);

        public byte[] Buffer { get; } = buffer;
        public int Length { get; } = length;
        public bool Pooled { get; } = pooled;
        public bool IsEmpty => Length <= 0 || Buffer.Length == 0;
    }

    private sealed class ParagraphEntry
    {
        public OcrBox Box { get; init; } = new();
        public string Text { get; init; } = "";
        public float Confidence { get; init; }
        public float MinX { get; init; }
        public float MaxX { get; init; }
        public float MinY { get; init; }
        public float MaxY { get; init; }
        public float Height { get; init; }
        public float CenterY { get; init; }
        public int GroupId { get; set; }
    }

    private static GrayBuffer RentGrayBuffer(int length)
    {
        if (length <= 0)
            return GrayBuffer.Empty;

        return new GrayBuffer(ByteArrayPool.Rent(length), length, pooled: true);
    }

    private static void ReturnGrayBuffer(GrayBuffer buffer)
    {
        if (buffer.Pooled && buffer.Buffer.Length > 0)
            ByteArrayPool.Return(buffer.Buffer);
    }

    private static float[] RentFloatBuffer(int length, bool clear = false)
    {
        float[] buffer = FloatArrayPool.Rent(length);
        if (clear)
            Array.Clear(buffer, 0, length);
        return buffer;
    }

    private static void ReturnFloatBuffer(float[]? buffer)
    {
        if (buffer != null)
            FloatArrayPool.Return(buffer);
    }

    private static int GetRecognitionResizedWidth(int width, int height, int targetHeight)
    {
        float ratio = width / (float)Math.Max(1, height);
        return Math.Max(RecogMinWidth, (int)Math.Ceiling(targetHeight * ratio));
    }

    private static int GetRecognitionResizedWidth(RecognitionSample sample, int targetHeight)
        => GetRecognitionResizedWidth(sample.Width, sample.Height, targetHeight);

    /// <summary>
    /// Detects and recognizes text in an image and returns the full structured OCR result.
    /// Supported recognizer languages are inferred automatically from ONNX metadata when available.
    /// </summary>
    /// <param name="detectorModelPath">Path to the CRAFT-compatible detector ONNX model.</param>
    /// <param name="recognizerModelPath">Path to the recognizer ONNX model.</param>
    /// <param name="imageResult">Input image to process.</param>
    /// <param name="enableWordBeamSearch">Enables dictionary-aware word beam search for segments backed by an available lexicon. Segments without a matching dictionary continue to use plain beam search.</param>
    /// <param name="allowRotatedRecognition">Enables rotation-based test-time augmentation for text regions. The default is <see langword="false"/> to reduce false positives on small boxes.</param>
    /// <param name="beamWidth">Beam width used by beam search.</param>
    /// <param name="dictionaryPath">Optional path to a custom dictionary file.</param>
    /// <returns>
    /// A <see cref="SeasonOcrResult"/> containing summary text, annotated image bytes,
    /// individual boxes, and paragraph-level grouping.
    /// </returns>
    public static SeasonOcrResult Detect(
        InferenceSession detectorSession,
        InferenceSession recognizerSession,
        ImageResult imageResult,
        bool enableWordBeamSearch = false,
        bool allowRotatedRecognition = false,
        bool createAnnotatedImage = false,
        int beamWidth = 5,
        string? dictionary = null)
    {
        var seasonOcrResult = new SeasonOcrResult();

        beamWidth = Math.Max(1, beamWidth);
        var rgb = ImageProcessor.ExtractRgb(imageResult);
        int srcW = imageResult.Width;
        int srcH = imageResult.Height;

        var boxes = DetectTextWithCraft(detectorSession, rgb, srcW, srcH);

        if (boxes.Count == 0)
        {
            seasonOcrResult.Summary = "No text detected.";

            if (createAnnotatedImage)
            {
                seasonOcrResult.AnnotatedImage = BuildFallbackImage(imageResult);
            }

            if (EnableDebugOutput)
            {
                seasonOcrResult.DebugReport = "detected_boxes=0\nrecognized_boxes=0\nparagraphs=0";
            }
        }
        else
        {
            string? modelCharset = null;

            int expectedClasses = 0;

            string recogInputName = recognizerSession.InputMetadata.Keys.First();

            string recogOutputName = recognizerSession.OutputMetadata.Keys.First();

            var outDims = recognizerSession.OutputMetadata[recogOutputName].Dimensions;

            for (int d = outDims.Length - 1; d >= 0; d--)
            {
                if (outDims[d] > 1)
                {
                    expectedClasses = outDims[d];
                    break;
                }
            }

            DebugLog($"[SeasonOCR] model output classes from metadata: {expectedClasses}");

            var charsetKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "character", "characters", "charset", "alphabet", "vocab", "chars", "labels"
            };

            foreach (var kv in recognizerSession.ModelMetadata.CustomMetadataMap)
            {
                DebugLog($"[SeasonOCR] model metadata: {kv.Key} = {(kv.Value.Length > 120 ? kv.Value.Substring(0, 120) + "..." : kv.Value)}");
                if (charsetKeys.Contains(kv.Key))
                    modelCharset = kv.Value;
            }

            if (modelCharset != null)
                DebugLog($"[SeasonOCR] model-embedded charset ({modelCharset.Length} chars)");

            modelCharset ??= BuildFallbackCharset(recognizerSession, recogOutputName);

            var recogInputDims = recognizerSession.InputMetadata[recogInputName].Dimensions;
            int recogHeight = RecogHeight;
            if (recogInputDims.Length >= 3 && recogInputDims[2] > 0)
                recogHeight = recogInputDims[2];
            DebugLog($"[SeasonOCR] recognizer input={recogInputName}, dims=[{string.Join(",", recogInputDims)}], height={recogHeight}");

            var recognizedLanguages = ParseLanguageMetadata(recognizerSession.ModelMetadata.CustomMetadataMap);
            if (recognizedLanguages.Count > 0)
            {
                DebugLog($"[SeasonOCR] recognizer languages from metadata=[{string.Join(",", recognizedLanguages)}]");
            }

            var decoderLexicon = LoadDecoderLexicon(
                recognizerSession.ModelMetadata.CustomMetadataMap,
                enableWordBeamSearch,
                dictionary,
                modelCharset,
                recognizedLanguages);

            RecognizeTextBatch(
                recognizerSession, recogInputName,
                rgb, srcW, srcH, boxes, recogHeight, modelCharset, enableWordBeamSearch, allowRotatedRecognition, beamWidth, decoderLexicon);

            var valid = new List<OcrBox>(boxes.Count);
            var includedInResult = new bool[boxes.Count];
            for (int i = 0; i < boxes.Count; i++)
            {
                if (string.IsNullOrWhiteSpace(boxes[i].Text))
                    continue;

                valid.Add(boxes[i]);
                includedInResult[i] = true;
            }

            string debugReport = null;

            if (valid.Count == 0)
            {
                seasonOcrResult.Boxes = boxes;

                seasonOcrResult.Summary = "Text detection succeeded, but recognition returned no result.";

                if (createAnnotatedImage)
                {
                    seasonOcrResult.AnnotatedImage = BuildFallbackImage(imageResult);
                }

                if (EnableDebugOutput)
                {
                    seasonOcrResult.DebugBoxes = BuildDebugBoxes(boxes, includedInResult);

                    seasonOcrResult.DebugReport = BuildDebugReport(boxes, null, []);
                }
            }
            else
            {
                seasonOcrResult.Boxes = valid;

                var paragraphs = BuildParagraphs(valid);

                seasonOcrResult.Summary = BuildSummary(valid, paragraphs);

                seasonOcrResult.Paragraphs = paragraphs;

                if (createAnnotatedImage)
                {
                    seasonOcrResult.AnnotatedImage = DrawResults(imageResult, valid);
                }

                if (EnableDebugOutput)
                {
                    var debugBoxes = BuildDebugBoxes(boxes, includedInResult);

                    debugReport = BuildDebugReport(boxes, debugBoxes, paragraphs);

                    DebugLog(debugReport);

                    seasonOcrResult.DebugReport = debugReport;
                }
            }
        }

        return seasonOcrResult;
    }
}
