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
    private static byte[] DrawResults(ImageResult image, List<OcrBox> boxes)
    {
        int pixelCount = image.Width * image.Height;
        var rgba = new byte[pixelCount * 4];
        for (int i = 0; i < pixelCount; i++)
        {
            int s = i * 3;
            int d = i * 4;
            rgba[d] = image.Data[s];
            rgba[d + 1] = image.Data[s + 1];
            rgba[d + 2] = image.Data[s + 2];
            rgba[d + 3] = 255;
        }

        int thickness = Math.Max(2, Math.Min(image.Width, image.Height) / 400);

        foreach (var box in boxes)
        {
            if (box.Quad is { Length: 4 } && IsFreeFormQuad(box.Quad))
            {
                DrawQuadOutline(rgba, image.Width, image.Height, box.Quad, thickness, 0, 255, 0);
            }
            else
            {
                DrawRectOutline(rgba, image.Width, image.Height,
                    box.X, box.Y, box.Width, box.Height, thickness, 0, 255, 0);
            }
        }

        var annotated = new ImageResult
        {
            Width = image.Width,
            Height = image.Height,
            SourceComp = ColorComponents.RedGreenBlueAlpha,
            Data = rgba
        };
        return ImageCodec.SaveAsJpeg(annotated, quality: 90);
    }

    private static void DrawRectOutline(
        byte[] pixels, int w, int h,
        int x, int y, int bw, int bh,
        int thickness, byte r, byte g, byte b)
    {
        int xmax = x + bw - 1;
        int ymax = y + bh - 1;
        for (int t = 0; t < thickness; t++)
        {
            for (int dx = x; dx <= xmax; dx++)
            {
                SetPixel(pixels, w, h, dx, y + t, r, g, b, 255);
                SetPixel(pixels, w, h, dx, ymax - t, r, g, b, 255);
            }
            for (int dy = y; dy <= ymax; dy++)
            {
                SetPixel(pixels, w, h, x + t, dy, r, g, b, 255);
                SetPixel(pixels, w, h, xmax - t, dy, r, g, b, 255);
            }
        }
    }

    private static void DrawQuadOutline(
        byte[] pixels, int w, int h, Vector2[] quad,
        int thickness, byte r, byte g, byte b)
    {
        for (int i = 0; i < 4; i++)
            DrawLine(pixels, w, h, quad[i], quad[(i + 1) % 4], thickness, r, g, b);
    }

    private static void DrawLine(
        byte[] pixels, int w, int h, Vector2 start, Vector2 end,
        int thickness, byte r, byte g, byte b)
    {
        float dx = end.X - start.X;
        float dy = end.Y - start.Y;
        int steps = Math.Max(1, (int)MathF.Ceiling(MathF.Max(MathF.Abs(dx), MathF.Abs(dy))));
        for (int i = 0; i <= steps; i++)
        {
            float t = i / (float)steps;
            int x = (int)MathF.Round(start.X + dx * t);
            int y = (int)MathF.Round(start.Y + dy * t);
            for (int oy = -thickness / 2; oy <= thickness / 2; oy++)
            {
                for (int ox = -thickness / 2; ox <= thickness / 2; ox++)
                    SetPixel(pixels, w, h, x + ox, y + oy, r, g, b, 255);
            }
        }
    }

    private static void SetPixel(byte[] pixels, int w, int h,
        int x, int y, byte r, byte g, byte b, byte a)
    {
        if ((uint)x >= w || (uint)y >= h) return;
        int idx = (y * w + x) * 4;
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = a;
    }

    private static string BuildSummary(List<OcrBox> boxes, List<OcrParagraph> paragraphs)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Detected {boxes.Count} text regions grouped into {paragraphs.Count} paragraphs:");
        for (int i = 0; i < paragraphs.Count; i++)
        {
            var paragraph = paragraphs[i];
            sb.AppendLine($"{i + 1}. {paragraph.Confidence:P0} \"{paragraph.Text}\"");
        }
        return sb.ToString().TrimEnd();
    }

    private static List<OcrDebugBox> BuildDebugBoxes(List<OcrBox> allBoxes, bool[] includedInResult)
    {
        var debugBoxes = new List<OcrDebugBox>(allBoxes.Count);
        for (int i = 0; i < allBoxes.Count; i++)
        {
            var box = allBoxes[i];
            debugBoxes.Add(new OcrDebugBox
            {
                Index = i,
                X = box.X,
                Y = box.Y,
                Width = box.Width,
                Height = box.Height,
                Quad = box.Quad.ToArray(),
                Text = box.Text,
                Confidence = box.Confidence,
                WinningAngle = box.DebugWinningAngle >= 0 ? box.DebugWinningAngle : null,
                RotationPolicy = box.DebugRotationPolicy,
                IncludedInResult = i < includedInResult.Length && includedInResult[i]
            });
        }

        return debugBoxes;
    }

    private static string BuildDebugReport(List<OcrBox> allBoxes, List<OcrDebugBox> debugBoxes, List<OcrParagraph> paragraphs)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"detected_boxes={allBoxes.Count}");
        sb.AppendLine($"recognized_boxes={debugBoxes.Count(box => box.IncludedInResult)}");
        sb.AppendLine($"paragraphs={paragraphs.Count}");
        sb.AppendLine();
        sb.AppendLine("[boxes]");

        foreach (var box in debugBoxes)
        {
            string status = box.IncludedInResult ? "keep" : "drop";
            sb.AppendLine(
                $"#{box.Index:D3} {status} conf={box.Confidence:F4} angle={(box.WinningAngle?.ToString() ?? "?")} tta={EscapeDebugText(box.RotationPolicy)} rect=[{box.X},{box.Y},{box.Width},{box.Height}] text=\"{EscapeDebugText(box.Text)}\"");
        }

        if (paragraphs.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("[paragraphs]");
            for (int i = 0; i < paragraphs.Count; i++)
            {
                var paragraph = paragraphs[i];
                sb.AppendLine(
                    $"#{i:D3} conf={paragraph.Confidence:F4} rect=[{paragraph.X},{paragraph.Y},{paragraph.Width},{paragraph.Height}] text=\"{EscapeDebugText(paragraph.Text)}\"");
            }
        }

        return sb.ToString().TrimEnd();
    }

    private static string EscapeDebugText(string? text)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;

        return text
            .Replace("\\", "\\\\", StringComparison.Ordinal)
            .Replace("\r", "\\r", StringComparison.Ordinal)
            .Replace("\n", "\\n", StringComparison.Ordinal)
            .Replace("\"", "\\\"", StringComparison.Ordinal);
    }

    private static List<OcrParagraph> BuildParagraphs(List<OcrBox> boxes)
    {
        var entries = new List<ParagraphEntry>(boxes.Count);
        foreach (var box in boxes)
        {
            if (string.IsNullOrWhiteSpace(box.Text))
                continue;

            Vector2[] quad = EnsureQuad(box);
            float minX = quad.Min(p => p.X);
            float maxX = quad.Max(p => p.X);
            float minY = quad.Min(p => p.Y);
            float maxY = quad.Max(p => p.Y);
            entries.Add(new ParagraphEntry
            {
                Box = box,
                Text = box.Text.Trim(),
                Confidence = box.Confidence,
                MinX = minX,
                MaxX = maxX,
                MinY = minY,
                MaxY = maxY,
                Height = Math.Max(1f, maxY - minY),
                CenterY = 0.5f * (minY + maxY),
                GroupId = 0
            });
        }

        if (entries.Count == 0)
            return [];

        entries.Sort((a, b) =>
        {
            int byY = a.MinY.CompareTo(b.MinY);
            return byY != 0 ? byY : a.MinX.CompareTo(b.MinX);
        });

        int currentGroup = 1;
        for (int seedIndex = 0; seedIndex < entries.Count; seedIndex++)
        {
            if (entries[seedIndex].GroupId != 0)
                continue;

            ParagraphEntry seed = entries[seedIndex];
            seed.GroupId = currentGroup;
            float minGx = seed.MinX;
            float maxGx = seed.MaxX;
            float minGy = seed.MinY;
            float maxGy = seed.MaxY;
            float totalHeight = seed.Height;
            int groupCount = 1;

            bool added;
            do
            {
                added = false;
                float meanHeight = totalHeight / groupCount;
                float expandedMinGx = minGx - ParagraphXThreshold * meanHeight;
                float expandedMaxGx = maxGx + ParagraphXThreshold * meanHeight;
                float expandedMinGy = minGy - ParagraphYThreshold * meanHeight;
                float expandedMaxGy = maxGy + ParagraphYThreshold * meanHeight;

                for (int i = seedIndex + 1; i < entries.Count; i++)
                {
                    ParagraphEntry entry = entries[i];
                    if (entry.GroupId != 0)
                        continue;

                    bool sameHorizontalLevel = (expandedMinGx <= entry.MinX && entry.MinX <= expandedMaxGx) ||
                                               (expandedMinGx <= entry.MaxX && entry.MaxX <= expandedMaxGx);
                    bool sameVerticalLevel = (expandedMinGy <= entry.MinY && entry.MinY <= expandedMaxGy) ||
                                             (expandedMinGy <= entry.MaxY && entry.MaxY <= expandedMaxGy);
                    if (!sameHorizontalLevel || !sameVerticalLevel)
                        continue;

                    entry.GroupId = currentGroup;
                    totalHeight += entry.Height;
                    groupCount++;
                    minGx = Math.Min(minGx, entry.MinX);
                    maxGx = Math.Max(maxGx, entry.MaxX);
                    minGy = Math.Min(minGy, entry.MinY);
                    maxGy = Math.Max(maxGy, entry.MaxY);
                    added = true;
                }
            }
            while (added);

            currentGroup++;
        }

        var groupedEntries = new SortedDictionary<int, List<ParagraphEntry>>();
        foreach (var entry in entries)
        {
            if (!groupedEntries.TryGetValue(entry.GroupId, out var group))
            {
                group = new List<ParagraphEntry>();
                groupedEntries[entry.GroupId] = group;
            }

            group.Add(entry);
        }

        var paragraphs = new List<OcrParagraph>();
        foreach (var pair in groupedEntries)
        {
            var current = pair.Value;
            float totalHeight = 0f;
            float totalConfidence = 0f;
            float minX = float.MaxValue;
            float maxX = float.MinValue;
            float minY = float.MaxValue;
            float maxY = float.MinValue;
            foreach (var entry in current)
            {
                totalHeight += entry.Height;
                totalConfidence += entry.Confidence;
                minX = Math.Min(minX, entry.MinX);
                maxX = Math.Max(maxX, entry.MaxX);
                minY = Math.Min(minY, entry.MinY);
                maxY = Math.Max(maxY, entry.MaxY);
            }

            float meanHeight = totalHeight / current.Count;
            float avgConfidence = totalConfidence / current.Count;
            current.Sort((a, b) =>
            {
                int byCenterY = a.CenterY.CompareTo(b.CenterY);
                return byCenterY != 0 ? byCenterY : a.MinX.CompareTo(b.MinX);
            });

            var words = new List<string>(current.Count);
            int start = 0;
            while (start < current.Count)
            {
                float lineLimit = current[start].CenterY + ParagraphLineThreshold * meanHeight;
                int end = start + 1;
                while (end < current.Count && current[end].CenterY < lineLimit)
                    end++;

                var lineEntries = new List<ParagraphEntry>(end - start);
                for (int i = start; i < end; i++)
                    lineEntries.Add(current[i]);

                lineEntries.Sort((a, b) =>
                {
                    int byX = a.MinX.CompareTo(b.MinX);
                    return byX != 0 ? byX : a.CenterY.CompareTo(b.CenterY);
                });

                foreach (var entry in lineEntries)
                    words.Add(entry.Text);

                start = end;
            }

            Vector2[] quad =
            [
                new Vector2(minX, minY),
                new Vector2(maxX, minY),
                new Vector2(maxX, maxY),
                new Vector2(minX, maxY)
            ];
            int x = Math.Max(0, (int)MathF.Floor(minX));
            int y = Math.Max(0, (int)MathF.Floor(minY));
            int w = Math.Max(1, (int)MathF.Ceiling(maxX) - x);
            int h = Math.Max(1, (int)MathF.Ceiling(maxY) - y);
            paragraphs.Add(new OcrParagraph
            {
                X = x,
                Y = y,
                Width = w,
                Height = h,
                Quad = quad,
                Text = string.Join(" ", words),
                Confidence = avgConfidence
            });
        }

        return paragraphs;
    }

    private static byte[] BuildFallbackImage(ImageResult image)
    {
        int pixelCount = image.Width * image.Height;
        var rgba = new byte[pixelCount * 4];
        for (int i = 0; i < pixelCount; i++)
        {
            int s = i * 3;
            int d = i * 4;
            rgba[d] = image.Data[s];
            rgba[d + 1] = image.Data[s + 1];
            rgba[d + 2] = image.Data[s + 2];
            rgba[d + 3] = 255;
        }
        var annotated = new ImageResult
        {
            Width = image.Width,
            Height = image.Height,
            SourceComp = ColorComponents.RedGreenBlueAlpha,
            Data = rgba
        };
        return ImageCodec.SaveAsJpeg(annotated, quality: 90);
    }
}
