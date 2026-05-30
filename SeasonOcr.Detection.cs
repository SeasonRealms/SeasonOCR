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
    private static List<OcrBox> DetectTextWithCraft(InferenceSession detectorSession, byte[] rgb, int srcW, int srcH)
    {
        // Preprocess with aspect-preserving resize to a 32-aligned canvas and RGB normalization.
        (float[] chw, int chwLength, int resizedW, int resizedH, float scale) =
            PreprocessCraft(rgb, srcW, srcH);
        try
        {
            var input = new DenseTensor<float>(
                new Memory<float>(chw, 0, chwLength),
                new[] { 1, 3, resizedH, resizedW });

            string inputName = detectorSession.InputMetadata.Keys.First();

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, input)
            };

            using var results = detectorSession.Run(inputs);
            var outputs = results.ToArray();

            DebugLog($"[SeasonOCR] CRAFT input={resizedW}x{resizedH}, model has {outputs.Length} output(s)");

            var (textScoreMap, linkScoreMap, outW, outH) = ExtractCraftScoreMaps(outputs);

            // Apply EasyOCR-style dual-channel CRAFT post-processing.
            var rawBoxes = ExtractBoxes(textScoreMap, linkScoreMap, outW, outH);

            // Map score-map coordinates back to source-image coordinates.
            float mapToInputX = (float)resizedW / outW;
            float mapToInputY = (float)resizedH / outH;
            float invScale = 1f / scale;

            DebugLog($"[SeasonOCR] mapToInput=({mapToInputX:F3},{mapToInputY:F3}), invScale={invScale:F4}");

            var boxes = new List<OcrBox>();
            foreach (var quad in rawBoxes)
            {
                var mappedQuad = new Vector2[quad.Length];
                for (int i = 0; i < quad.Length; i++)
                {
                    float qx = quad[i].X * mapToInputX * invScale;
                    float qy = quad[i].Y * mapToInputY * invScale;
                    mappedQuad[i] = new Vector2(
                        Math.Clamp(qx, 0, srcW - 1),
                        Math.Clamp(qy, 0, srcH - 1));
                }

                (int ox, int oy, int ow, int oh) = GetBoundingRect(mappedQuad, srcW, srcH);
                if (ow * oh >= MinBoxArea)
                    boxes.Add(new OcrBox { X = ox, Y = oy, Width = ow, Height = oh, Quad = mappedQuad });
            }

            DebugLog($"[SeasonOCR] raw boxes={rawBoxes.Count}, after merge/filter={boxes.Count}");

            return GroupTextRegions(boxes, srcW, srcH);
        }
        finally
        {
            ReturnFloatBuffer(chw);
        }
    }

    private static (float[] textScoreMap, float[] linkScoreMap, int outW, int outH) ExtractCraftScoreMaps(
        DisposableNamedOnnxValue[] outputs)
    {
        float[]? bestText = null;
        float[]? bestLink = null;
        int bestW = 0, bestH = 0;

        for (int i = 0; i < outputs.Length; i++)
        {
            Tensor<float> tensor;
            try
            {
                tensor = outputs[i].AsTensor<float>();
            }
            catch
            {
                continue;
            }

            var dims = tensor.Dimensions.ToArray();
            DebugLog($"[SeasonOCR]   output[{i}] dims=[{string.Join(",", dims)}]");

            if (!TryExtractCraftScoreMaps(tensor, out var textMap, out var linkMap, out var outW, out var outH))
                continue;

            int area = outW * outH;
            if (area <= bestW * bestH)
                continue;

            bestText = NormalizeCraftScoreMap(textMap, "text");
            bestLink = NormalizeCraftScoreMap(linkMap, "link");
            bestW = outW;
            bestH = outH;
            DebugLog($"[SeasonOCR]   selected output[{i}] as score maps: {outW}x{outH}");
        }

        if (bestText == null || bestLink == null)
            throw new InvalidOperationException("Failed to extract text/link score maps from the CRAFT ONNX outputs.");

        return (bestText!, bestLink!, bestW, bestH);
    }

    private static bool TryExtractCraftScoreMaps(
        Tensor<float> tensor,
        out float[] textScoreMap,
        out float[] linkScoreMap,
        out int outW,
        out int outH)
    {
        var dims = tensor.Dimensions.ToArray();
        textScoreMap = [];
        linkScoreMap = [];
        outW = 0;
        outH = 0;

        if (dims.Length == 4 && dims[1] == 2)
        {
            outH = dims[2];
            outW = dims[3];
            textScoreMap = new float[outW * outH];
            linkScoreMap = new float[outW * outH];
            for (int y = 0; y < outH; y++)
            {
                for (int x = 0; x < outW; x++)
                {
                    int idx = y * outW + x;
                    textScoreMap[idx] = tensor[0, 0, y, x];
                    linkScoreMap[idx] = tensor[0, 1, y, x];
                }
            }
            return true;
        }

        if (dims.Length == 4 && dims[^1] == 2)
        {
            outH = dims[1];
            outW = dims[2];
            textScoreMap = new float[outW * outH];
            linkScoreMap = new float[outW * outH];
            for (int y = 0; y < outH; y++)
            {
                for (int x = 0; x < outW; x++)
                {
                    int idx = y * outW + x;
                    textScoreMap[idx] = tensor[0, y, x, 0];
                    linkScoreMap[idx] = tensor[0, y, x, 1];
                }
            }
            return true;
        }

        if (dims.Length == 3 && dims[0] == 2)
        {
            outH = dims[1];
            outW = dims[2];
            textScoreMap = new float[outW * outH];
            linkScoreMap = new float[outW * outH];
            for (int y = 0; y < outH; y++)
            {
                for (int x = 0; x < outW; x++)
                {
                    int idx = y * outW + x;
                    textScoreMap[idx] = tensor[0, y, x];
                    linkScoreMap[idx] = tensor[1, y, x];
                }
            }
            return true;
        }

        if (dims.Length == 3 && dims[^1] == 2)
        {
            outH = dims[0];
            outW = dims[1];
            textScoreMap = new float[outW * outH];
            linkScoreMap = new float[outW * outH];
            for (int y = 0; y < outH; y++)
            {
                for (int x = 0; x < outW; x++)
                {
                    int idx = y * outW + x;
                    textScoreMap[idx] = tensor[y, x, 0];
                    linkScoreMap[idx] = tensor[y, x, 1];
                }
            }
            return true;
        }

        return false;
    }

    private static float[] NormalizeCraftScoreMap(float[] rawMap, string name)
    {
        float minVal = rawMap.Min();
        float maxVal = rawMap.Max();
        bool alreadyNormalized = minVal >= -0.1f && maxVal <= 1.2f;
        DebugLog($"[SeasonOCR] {name}Map raw range=[{minVal:F4}, {maxVal:F4}] sigmoidSkipped={alreadyNormalized}");

        var normalized = new float[rawMap.Length];
        for (int i = 0; i < rawMap.Length; i++)
            normalized[i] = alreadyNormalized ? Math.Clamp(rawMap[i], 0f, 1f) : Sigmoid(rawMap[i]);
        return normalized;
    }

    /// <summary>
    /// CRAFT preprocessing with aspect-preserving resize to a 32-aligned canvas
    /// and ImageNet mean/std normalization.
    /// </summary>
    private static (float[] chw, int chwLength, int w, int h, float scale) PreprocessCraft(
        byte[] rgb, int srcW, int srcH)
    {
        const int alignment = 32;

        float targetSize = Math.Min(CraftCanvasSize, CraftMagRatio * Math.Max(srcW, srcH));
        float scale = targetSize / Math.Max(srcW, srcH);
        int newW = Math.Max(1, (int)(srcW * scale));
        int newH = Math.Max(1, (int)(srcH * scale));

        // Align to a multiple of 32.
        int padW = ((newW + alignment - 1) / alignment) * alignment;
        int padH = ((newH + alignment - 1) / alignment) * alignment;

        float actualScale = (float)newW / srcW; // Effective resize scale.

        // ImageNet RGB mean and standard deviation.
        var mean = new[] { 0.485f, 0.456f, 0.406f };
        var std = new[] { 0.229f, 0.224f, 0.225f };

        // Resize first, then pad.
        int resizedLength = 3 * newW * newH;
        float[] resized = RentFloatBuffer(resizedLength);
        ImageProcessor.BilinearResizeInto(rgb, srcW, srcH, newW, newH, resized);

        // Pad to padW x padH with zeros and apply ImageNet normalization.
        int chwLength = 3 * padW * padH;
        float[] result = RentFloatBuffer(chwLength, clear: true);
        try
        {
            int chPlane = padW * padH;
            for (int c = 0; c < 3; c++)
            {
                int dstCh = c * chPlane;
                int srcCh = c * newW * newH;
                for (int y = 0; y < newH; y++)
                {
                    int dstRow = y * padW;
                    int srcRow = y * newW;
                    for (int x = 0; x < newW; x++)
                    {
                        float val = resized[srcCh + srcRow + x]; // Already divided by 255 into [0, 1].
                        result[dstCh + dstRow + x] = (val - mean[c]) / std[c];
                    }
                }
            }

            return (result, chwLength, padW, padH, actualScale);
        }
        finally
        {
            ReturnFloatBuffer(resized);
        }
    }

    /// <summary>
    /// Extracts text regions from the region score map.
    /// Uses EasyOCR-style connected components over combined text/link scores.
    /// Returns four-point boxes in score-map coordinates.
    /// </summary>
    private static List<Vector2[]> ExtractBoxes(
        float[] textScoreMap, float[] linkScoreMap, int mapW, int mapH)
    {
        int total = mapW * mapH;
        var textMask = new byte[total];
        var linkMask = new byte[total];
        var combinedMask = new byte[total];
        int combinedCount = 0;
        for (int i = 0; i < total; i++)
        {
            if (textScoreMap[i] >= TextThreshLow)
            {
                textMask[i] = 1;
                combinedMask[i] = 1;
            }
            if (linkScoreMap[i] >= LinkThresh)
            {
                linkMask[i] = 1;
                combinedMask[i] = 1;
            }
            if (combinedMask[i] != 0)
                combinedCount++;
        }

        DebugLog($"[SeasonOCR] craft foreground pixels={combinedCount}/{total}");

        if (combinedCount == 0)
            return new List<Vector2[]>();

        var boxes = new List<Vector2[]>();
        var visited = new byte[total];
        var queue = new Queue<int>();

        for (int i = 0; i < total; i++)
        {
            if (combinedMask[i] == 0 || visited[i] != 0)
                continue;

            queue.Clear();
            queue.Enqueue(i);
            visited[i] = 1;

            var componentPixels = new List<int>();
            int minX = mapW, minY = mapH, maxX = 0, maxY = 0;
            float maxTextScore = 0f;

            while (queue.Count > 0)
            {
                int idx = queue.Dequeue();
                componentPixels.Add(idx);
                int px = idx % mapW;
                int py = idx / mapW;
                if (px < minX) minX = px;
                if (py < minY) minY = py;
                if (px > maxX) maxX = px;
                if (py > maxY) maxY = py;
                if (textScoreMap[idx] > maxTextScore) maxTextScore = textScoreMap[idx];

                TryEnqueue(px - 1, py);
                TryEnqueue(px + 1, py);
                TryEnqueue(px, py - 1);
                TryEnqueue(px, py + 1);
            }

            int componentSize = componentPixels.Count;
            if (componentSize < MinComponentSize || maxTextScore < TextThreshHigh)
                continue;

            var segPixels = new List<Vector2>();
            foreach (int idx in componentPixels)
            {
                if (linkMask[idx] != 0 && textMask[idx] == 0)
                    continue;

                int px = idx % mapW;
                int py = idx / mapW;
                segPixels.Add(new Vector2(px, py));
            }

            if (segPixels.Count == 0)
                continue;

            int segMinX = (int)segPixels.Min(p => p.X);
            int segMinY = (int)segPixels.Min(p => p.Y);
            int segMaxX = (int)segPixels.Max(p => p.X);
            int segMaxY = (int)segPixels.Max(p => p.Y);
            int compW = Math.Max(1, segMaxX - segMinX + 1);
            int compH = Math.Max(1, segMaxY - segMinY + 1);
            int niter = (int)(Math.Sqrt(componentSize * Math.Min(compW, compH) / (float)(compW * compH)) * 2f);
            int margin = ComputeComponentMargin(componentSize, compW, compH, niter);
            Vector2[] quad = BuildComponentQuad(segPixels, margin, mapW, mapH);
            (int bx, int by, int bw, int bh) = GetBoundingRect(quad, mapW, mapH);
            if (bw > 2 && bh > 2)
                boxes.Add(quad);
        }

        return boxes;

        void TryEnqueue(int x, int y)
        {
            if ((uint)x >= mapW || (uint)y >= mapH)
                return;

            int idx = y * mapW + x;
            if (combinedMask[idx] == 0 || visited[idx] != 0)
                return;

            visited[idx] = 1;
            queue.Enqueue(idx);
        }
    }

    /// <summary>
    /// Groups detection results with logic similar to EasyOCR's group_text_box,
    /// splitting results into horizontal and rotated boxes.
    /// </summary>
    private static List<OcrBox> GroupTextRegions(List<OcrBox> boxes, int srcW, int srcH)
    {
        if (boxes.Count <= 1)
            return boxes;

        var horizontal = new List<(float xMin, float xMax, float yMin, float yMax, float centerY, float height)>();
        var free = new List<OcrBox>();

        foreach (var box in boxes)
        {
            Vector2[] quad = EnsureQuad(box);
            float slopeUp = (quad[1].Y - quad[0].Y) / Math.Max(10f, quad[1].X - quad[0].X);
            float slopeDown = (quad[2].Y - quad[3].Y) / Math.Max(10f, quad[2].X - quad[3].X);
            if (Math.Max(Math.Abs(slopeUp), Math.Abs(slopeDown)) < GroupSlopeThreshold)
            {
                float xMin = quad.Min(p => p.X);
                float xMax = quad.Max(p => p.X);
                float yMin = quad.Min(p => p.Y);
                float yMax = quad.Max(p => p.Y);
                horizontal.Add((xMin, xMax, yMin, yMax, 0.5f * (yMin + yMax), yMax - yMin));
            }
            else
            {
                Vector2[] expanded = ExpandFreeQuad(quad, GroupAddMargin, srcW, srcH);
                var freeBox = CreateBoxFromQuad(expanded, srcW, srcH);
                if (freeBox.Width * freeBox.Height >= MinBoxArea)
                    free.Add(freeBox);
            }
        }

        horizontal.Sort((a, b) => a.centerY.CompareTo(b.centerY));
        var combinedLines = new List<List<(float xMin, float xMax, float yMin, float yMax, float centerY, float height)>>();
        var currentLine = new List<(float xMin, float xMax, float yMin, float yMax, float centerY, float height)>();
        var lineHeights = new List<float>();
        var lineCenters = new List<float>();

        foreach (var box in horizontal)
        {
            if (currentLine.Count == 0)
            {
                currentLine.Add(box);
                lineHeights.Add(box.height);
                lineCenters.Add(box.centerY);
            }
            else if (ShouldStayOnSameLine(currentLine, box))
            {
                currentLine.Add(box);
                lineHeights.Add(box.height);
                lineCenters.Add(box.centerY);
            }
            else
            {
                combinedLines.Add(currentLine);
                currentLine = [box];
                lineHeights = [box.height];
                lineCenters = [box.centerY];
            }
        }

        if (currentLine.Count > 0)
            combinedLines.Add(currentLine);

        var merged = new List<OcrBox>();
        foreach (var line in combinedLines)
        {
            var ordered = line.OrderBy(b => b.xMin).ToList();
            var groups = new List<List<(float xMin, float xMax, float yMin, float yMax, float centerY, float height)>>();
            var currentGroup = new List<(float xMin, float xMax, float yMin, float yMax, float centerY, float height)>();
            var heights = new List<float>();
            float currentXMax = 0;

            foreach (var box in ordered)
            {
                if (currentGroup.Count == 0)
                {
                    currentGroup.Add(box);
                    heights.Add(box.height);
                    currentXMax = box.xMax;
                    continue;
                }

                var previous = currentGroup[^1];
                float avgHeight = heights.Average();
                float gap = box.xMin - currentXMax;
                bool compatibleHeight = Math.Abs(avgHeight - box.height) < GroupHeightThreshold * avgHeight;
                bool closeEnough = gap < GroupWidthThreshold * (box.yMax - box.yMin);
                bool keepSeparate = ShouldKeepBoxesSeparate(previous, box);

                if (compatibleHeight && closeEnough && !keepSeparate)
                {
                    currentGroup.Add(box);
                    heights.Add(box.height);
                    currentXMax = box.xMax;
                }
                else
                {
                    groups.Add(currentGroup);
                    currentGroup = [box];
                    heights = [box.height];
                    currentXMax = box.xMax;
                }
            }

            if (currentGroup.Count > 0)
                groups.Add(currentGroup);

            foreach (var group in groups)
                merged.Add(BuildMergedHorizontalBox(group, srcW, srcH));
        }

        merged.AddRange(free);
        return merged
            .OrderBy(b => b.Quad.Length == 4 ? b.Quad[0].Y : b.Y)
            .ThenBy(b => b.Quad.Length == 4 ? b.Quad[0].X : b.X)
            .ToList();
    }

    private static int ComputeComponentMargin(int componentSize, int compW, int compH, int niter)
    {
        int margin = Math.Max(1, niter);
        int maxSide = Math.Max(compW, compH);
        int minSide = Math.Min(compW, compH);
        bool compactComponent = maxSide <= 28 || componentSize <= 80;
        bool nearlySquare = maxSide <= minSide * 2;

        // Compact sign glyphs are prone to sticking together when expanded too aggressively.
        if (compactComponent && nearlySquare)
            margin = Math.Max(1, (int)Math.Ceiling(niter * 0.5f));

        return margin;
    }

    private static bool ShouldStayOnSameLine(
        List<(float xMin, float xMax, float yMin, float yMax, float centerY, float height)> currentLine,
        (float xMin, float xMax, float yMin, float yMax, float centerY, float height) candidate)
    {
        float avgCenter = currentLine.Average(b => b.centerY);
        float avgHeight = currentLine.Average(b => b.height);
        if (Math.Abs(avgCenter - candidate.centerY) >= GroupYCenterThreshold * avgHeight)
            return false;

        // If a compact box vertically stacks on top of an existing compact box with strong x-overlap,
        // force a new line to avoid merging sign columns like "东/309" or "西/315/W".
        foreach (var existing in currentLine)
        {
            if (!IsCompactBox(existing) || !IsCompactBox(candidate))
                continue;

            float overlap = ComputeHorizontalOverlapRatio(existing, candidate);
            float verticalGap = ComputeVerticalGap(existing, candidate);
            if (overlap >= 0.45f && verticalGap > 0f)
                return false;
        }

        return true;
    }

    private static bool ShouldKeepBoxesSeparate(
        (float xMin, float xMax, float yMin, float yMax, float centerY, float height) left,
        (float xMin, float xMax, float yMin, float yMax, float centerY, float height) right)
    {
        if (!IsCompactBox(left) || !IsCompactBox(right))
            return false;

        float overlap = ComputeHorizontalOverlapRatio(left, right);
        if (overlap >= 0.35f)
            return true;

        float gap = right.xMin - left.xMax;
        float meanHeight = 0.5f * (left.height + right.height);
        float widthLeft = left.xMax - left.xMin;
        float widthRight = right.xMax - right.xMin;
        bool bothNarrow = widthLeft <= meanHeight * 1.8f && widthRight <= meanHeight * 1.8f;
        if (bothNarrow && gap > meanHeight * 0.6f)
            return true;

        return false;
    }

    private static bool IsCompactBox((float xMin, float xMax, float yMin, float yMax, float centerY, float height) box)
    {
        float width = box.xMax - box.xMin;
        float height = Math.Max(1f, box.yMax - box.yMin);
        float area = width * height;
        float ratio = width / height;
        return area <= 12000f && ratio <= 2.2f;
    }

    private static float ComputeHorizontalOverlapRatio(
        (float xMin, float xMax, float yMin, float yMax, float centerY, float height) a,
        (float xMin, float xMax, float yMin, float yMax, float centerY, float height) b)
    {
        float overlap = Math.Max(0f, Math.Min(a.xMax, b.xMax) - Math.Max(a.xMin, b.xMin));
        float minWidth = Math.Max(1f, Math.Min(a.xMax - a.xMin, b.xMax - b.xMin));
        return overlap / minWidth;
    }

    private static float ComputeVerticalGap(
        (float xMin, float xMax, float yMin, float yMax, float centerY, float height) a,
        (float xMin, float xMax, float yMin, float yMax, float centerY, float height) b)
    {
        if (b.yMin >= a.yMax)
            return b.yMin - a.yMax;
        if (a.yMin >= b.yMax)
            return a.yMin - b.yMax;
        return 0f;
    }

    private static OcrBox BuildMergedHorizontalBox(
        List<(float xMin, float xMax, float yMin, float yMax, float centerY, float height)> group,
        int imageW,
        int imageH)
    {
        float minX = group.Min(b => b.xMin);
        float maxX = group.Max(b => b.xMax);
        float minY = group.Min(b => b.yMin);
        float maxY = group.Max(b => b.yMax);
        float boxWidth = Math.Max(1f, maxX - minX);
        float boxHeight = Math.Max(1f, maxY - minY);
        float margin = Math.Max(1f, GroupAddMargin * Math.Min(boxWidth, boxHeight));

        Vector2[] quad =
        [
            new Vector2(minX - margin, minY - margin),
            new Vector2(maxX + margin, minY - margin),
            new Vector2(maxX + margin, maxY + margin),
            new Vector2(minX - margin, maxY + margin)
        ];
        return CreateBoxFromQuad(ClampQuad(quad, imageW, imageH), imageW, imageH);
    }

    private static Vector2[] BuildComponentQuad(List<Vector2> pixels, int margin, int mapW, int mapH)
    {
        if (pixels.Count == 0)
            return [];

        float meanX = 0f;
        float meanY = 0f;
        foreach (var p in pixels)
        {
            meanX += p.X;
            meanY += p.Y;
        }
        meanX /= pixels.Count;
        meanY /= pixels.Count;

        float covXX = 0f, covXY = 0f, covYY = 0f;
        foreach (var p in pixels)
        {
            float dx = p.X - meanX;
            float dy = p.Y - meanY;
            covXX += dx * dx;
            covXY += dx * dy;
            covYY += dy * dy;
        }

        float angle = 0.5f * MathF.Atan2(2f * covXY, covXX - covYY);
        Vector2 ux = new(MathF.Cos(angle), MathF.Sin(angle));
        Vector2 uy = new(-ux.Y, ux.X);

        float minU = float.MaxValue, maxU = float.MinValue;
        float minV = float.MaxValue, maxV = float.MinValue;
        foreach (var p in pixels)
        {
            Vector2 d = p - new Vector2(meanX, meanY);
            float u = Vector2.Dot(d, ux);
            float v = Vector2.Dot(d, uy);
            if (u < minU) minU = u;
            if (u > maxU) maxU = u;
            if (v < minV) minV = v;
            if (v > maxV) maxV = v;
        }

        minU -= margin;
        maxU += margin;
        minV -= margin;
        maxV += margin;

        Vector2 center = new(meanX, meanY);
        Vector2[] quad =
        [
            center + ux * minU + uy * minV,
            center + ux * maxU + uy * minV,
            center + ux * maxU + uy * maxV,
            center + ux * minU + uy * maxV
        ];
        return OrderQuad(ClampQuad(quad, mapW, mapH));
    }

    private static Vector2[] ExpandFreeQuad(Vector2[] quad, float addMargin, int srcW, int srcH)
    {
        quad = OrderQuad(quad);
        float height = Vector2.Distance(quad[3], quad[0]);
        float width = Vector2.Distance(quad[1], quad[0]);
        float margin = 1.44f * addMargin * MathF.Min(width, height);

        float theta13 = MathF.Abs(MathF.Atan2(quad[0].Y - quad[2].Y, MathF.Max(10f, quad[0].X - quad[2].X)));
        float theta24 = MathF.Abs(MathF.Atan2(quad[1].Y - quad[3].Y, MathF.Max(10f, quad[1].X - quad[3].X)));
        Vector2[] expanded =
        [
            new Vector2(quad[0].X - MathF.Cos(theta13) * margin, quad[0].Y - MathF.Sin(theta13) * margin),
            new Vector2(quad[1].X + MathF.Cos(theta24) * margin, quad[1].Y - MathF.Sin(theta24) * margin),
            new Vector2(quad[2].X + MathF.Cos(theta13) * margin, quad[2].Y + MathF.Sin(theta13) * margin),
            new Vector2(quad[3].X - MathF.Cos(theta24) * margin, quad[3].Y + MathF.Sin(theta24) * margin)
        ];
        return OrderQuad(ClampQuad(expanded, srcW, srcH));
    }

    private static Vector2[] EnsureQuad(OcrBox box)
    {
        if (box.Quad is { Length: 4 })
            return OrderQuad(box.Quad);

        return
        [
            new Vector2(box.X, box.Y),
            new Vector2(box.X + box.Width, box.Y),
            new Vector2(box.X + box.Width, box.Y + box.Height),
            new Vector2(box.X, box.Y + box.Height)
        ];
    }

    private static Vector2[] ClampQuad(Vector2[] quad, int width, int height)
    {
        var result = new Vector2[quad.Length];
        for (int i = 0; i < quad.Length; i++)
        {
            result[i] = new Vector2(
                Math.Clamp(quad[i].X, 0, Math.Max(0, width - 1)),
                Math.Clamp(quad[i].Y, 0, Math.Max(0, height - 1)));
        }
        return result;
    }

    private static Vector2[] OrderQuad(Vector2[] quad)
    {
        if (quad.Length != 4)
            return quad;

        var sorted = quad.OrderBy(p => p.Y).ThenBy(p => p.X).ToArray();
        var top = sorted.Take(2).OrderBy(p => p.X).ToArray();
        var bottom = sorted.Skip(2).OrderByDescending(p => p.X).ToArray();
        return [top[0], top[1], bottom[0], bottom[1]];
    }

    private static (int x, int y, int w, int h) GetBoundingRect(Vector2[] quad, int imageW, int imageH)
    {
        float minX = quad.Min(p => p.X);
        float maxX = quad.Max(p => p.X);
        float minY = quad.Min(p => p.Y);
        float maxY = quad.Max(p => p.Y);
        int x = Math.Max(0, (int)MathF.Floor(minX));
        int y = Math.Max(0, (int)MathF.Floor(minY));
        int w = Math.Min(imageW - x, Math.Max(1, (int)MathF.Ceiling(maxX) - x));
        int h = Math.Min(imageH - y, Math.Max(1, (int)MathF.Ceiling(maxY) - y));
        return (x, y, w, h);
    }

    private static OcrBox CreateBoxFromQuad(Vector2[] quad, int imageW, int imageH)
    {
        var ordered = OrderQuad(quad);
        (int x, int y, int w, int h) = GetBoundingRect(ordered, imageW, imageH);
        return new OcrBox { X = x, Y = y, Width = w, Height = h, Quad = ordered };
    }
}
