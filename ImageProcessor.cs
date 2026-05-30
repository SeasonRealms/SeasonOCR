// Copyright (c) SeasonEngine and contributors.
// Licensed under the Apache License, Version 2.0.
// SeasonOCR for EasyOCR Models

namespace SeasonOCR;

/// <summary>
/// Shared image preprocessing helpers for AI models.
/// Provides RGBA-to-RGB extraction, bilinear resizing, letterbox and center-crop
/// resizing, and normalization helpers.
/// </summary>
internal static class ImageProcessor
{
    /// <summary>
    /// Extracts an RGB byte array from an <see cref="ImageResult"/>.
    /// Uses the raw buffer length to distinguish RGB and RGBA input instead of relying
    /// on <see cref="ImageResult.SourceComp"/>, which may not always match the actual data.
    /// </summary>
    public static byte[] ExtractRgb(ImageResult image)
    {
        int pixelCount = image.Width * image.Height;

        // Infer the actual pixel format from the byte length.
        if (image.Data.Length == pixelCount * 3)
        {
            // Already RGB, so return a copy.
            var rgb = new byte[image.Data.Length];
            Array.Copy(image.Data, rgb, image.Data.Length);
            return rgb;
        }

        if (image.Data.Length == pixelCount * 4)
        {
            // Convert RGBA to RGB by dropping the alpha channel.
            var result = new byte[pixelCount * 3];
            for (int i = 0; i < pixelCount; i++)
            {
                int src = i * 4;
                int dst = i * 3;
                result[dst] = image.Data[src];         // R
                result[dst + 1] = image.Data[src + 1]; // G
                result[dst + 2] = image.Data[src + 2]; // B
            }
            return result;
        }

        throw new NotSupportedException(
            $"Unsupported image buffer length: {image.Data.Length}, pixel count: {pixelCount}, " +
            $"expected {pixelCount * 3} (RGB) or {pixelCount * 4} (RGBA).");
    }

    /// <summary>
    /// Resizes an RGB image with bilinear interpolation.
    /// Input format is HWC byte RGB; output format is NCHW float with values normalized by 255.
    /// </summary>
    public static float[] BilinearResize(byte[] srcRgb, int srcW, int srcH, int dstW, int dstH)
    {
        float[] result = new float[3 * dstW * dstH];
        BilinearResizeInto(srcRgb, srcW, srcH, dstW, dstH, result);
        return result;
    }

    /// <summary>
    /// Resizes an RGB image with bilinear interpolation into a caller-provided buffer.
    /// Input format is HWC byte RGB; output format is NCHW float with values normalized by 255.
    /// </summary>
    public static void BilinearResizeInto(byte[] srcRgb, int srcW, int srcH, int dstW, int dstH, float[] destination)
    {
        int requiredLength = 3 * dstW * dstH;
        if (destination.Length < requiredLength)
            throw new ArgumentException($"Destination buffer is too small. Required at least {requiredLength} elements.", nameof(destination));

        float scaleX = (float)srcW / dstW;
        float scaleY = (float)srcH / dstH;

        for (int dy = 0; dy < dstH; dy++)
        {
            float sy = dy * scaleY;
            int y0 = (int)sy;
            int y1 = Math.Min(y0 + 1, srcH - 1);
            float fy = sy - y0;

            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = dx * scaleX;
                int x0 = (int)sx;
                int x1 = Math.Min(x0 + 1, srcW - 1);
                float fx = sx - x0;

                int dstIdx = dy * dstW + dx;

                for (int c = 0; c < 3; c++)
                {
                    int srcIdx00 = (y0 * srcW + x0) * 3 + c;
                    int srcIdx10 = (y0 * srcW + x1) * 3 + c;
                    int srcIdx01 = (y1 * srcW + x0) * 3 + c;
                    int srcIdx11 = (y1 * srcW + x1) * 3 + c;

                    float v00 = srcRgb[srcIdx00];
                    float v10 = srcRgb[srcIdx10];
                    float v01 = srcRgb[srcIdx01];
                    float v11 = srcRgb[srcIdx11];

                    float top = (1 - fx) * v00 + fx * v10;
                    float bot = (1 - fx) * v01 + fx * v11;
                    float val = ((1 - fy) * top + fy * bot) / 255.0f;

                    destination[c * dstW * dstH + dstIdx] = val;
                }
            }
        }
    }

    /// <summary>
    /// Applies YOLO-style letterbox resize and normalization.
    /// The image keeps its aspect ratio and unused space is filled with gray (114 / 255).
    /// Returns NCHW float data.
    /// </summary>
    public static float[] LetterboxResizeNormalize(
        byte[] srcRgb, int srcW, int srcH, int dstW, int dstH)
    {
        // Compute the resize ratio while keeping the aspect ratio.
        float scale = Math.Min((float)dstW / srcW, (float)dstH / srcH);
        int newW = Math.Max(1, (int)(srcW * scale));
        int newH = Math.Max(1, (int)(srcH * scale));

        // Resize with bilinear interpolation.
        float[] resized = BilinearResize(srcRgb, srcW, srcH, newW, newH);

        // Gray padding value: 114 / 255 ~= 0.447.
        const float gray = 114.0f / 255.0f;
        float[] result = new float[3 * dstW * dstH];
        for (int i = 0; i < result.Length; i++)
            result[i] = gray;

        int padX = (dstW - newW) / 2;
        int padY = (dstH - newH) / 2;

        // Copy the resized image into the centered target region.
        for (int c = 0; c < 3; c++)
        {
            int chOffset = c * dstW * dstH;
            int srcChOffset = c * newW * newH;

            for (int y = 0; y < newH; y++)
            {
                int dstRow = (y + padY) * dstW + padX;
                int srcRow = y * newW;

                for (int x = 0; x < newW; x++)
                {
                    result[chOffset + dstRow + x] = resized[srcChOffset + srcRow + x];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Applies center-crop resize and ImageNet mean/std normalization in a ResNet-style pipeline.
    /// The shorter side is matched first and the result is then center-cropped to the target size.
    /// Returns NCHW float data.
    /// </summary>
    public static float[] CenterCropResizeNormalize(
        byte[] srcRgb, int srcW, int srcH, int dstW, int dstH,
        float[] mean, float[] stddev)
    {
        // Compute the resize ratio so the shorter side matches before cropping.
        float scale = Math.Max((float)dstW / srcW, (float)dstH / srcH);
        float intW = srcW * scale;
        float intH = srcH * scale;
        float cropX = (intW - dstW) / 2f;
        float cropY = (intH - dstH) / 2f;

        float invScale = 1f / scale;
        float[] result = new float[3 * dstW * dstH];

        for (int dy = 0; dy < dstH; dy++)
        {
            float sy = (cropY + dy) * invScale;
            int y0 = (int)sy;
            int y1 = Math.Min(y0 + 1, srcH - 1);
            float fy = sy - y0;

            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = (cropX + dx) * invScale;
                int x0 = (int)sx;
                int x1 = Math.Min(x0 + 1, srcW - 1);
                float fx = sx - x0;

                int dstIdx = dy * dstW + dx;

                for (int c = 0; c < 3; c++)
                {
                    int srcIdx00 = (y0 * srcW + x0) * 3 + c;
                    int srcIdx10 = (y0 * srcW + x1) * 3 + c;
                    int srcIdx01 = (y1 * srcW + x0) * 3 + c;
                    int srcIdx11 = (y1 * srcW + x1) * 3 + c;

                    float v00 = srcRgb[srcIdx00];
                    float v10 = srcRgb[srcIdx10];
                    float v01 = srcRgb[srcIdx01];
                    float v11 = srcRgb[srcIdx11];

                    float top = (1 - fx) * v00 + fx * v10;
                    float bot = (1 - fx) * v01 + fx * v11;
                    float val = ((1 - fy) * top + fy * bot) / 255.0f;

                    result[c * dstW * dstH + dstIdx] = (val - mean[c]) / stddev[c];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Catmull-Rom cubic kernel, equivalent to ImageSharp bicubic resampling.
    /// </summary>
    private static float Cubic(float x)
    {
        const float a = -0.5f;
        x = Math.Abs(x);
        if (x > 2.0f) return 0;
        float x2 = x * x;
        if (x <= 1.0f) return (a + 2) * x * x2 - (a + 3) * x2 + 1;
        return a * x * x2 - 5 * a * x2 + 8 * a * x - 4 * a;
    }

    /// <summary>
    /// Applies aspect-preserving resize, 32-aligned padding, BGR channel reordering,
    /// and mean normalization in a Faster R-CNN style pipeline.
    /// Resize uses Catmull-Rom bicubic sampling, matching torchvision/ImageSharp behavior.
    /// Input is HWC RGB bytes; output is CHW BGR floats without divide-by-255 or stddev scaling.
    /// Padding uses zero-filled pixels on the bottom and right edges.
    /// </summary>
    /// <param name="srcRgb">Source RGB byte buffer.</param>
    /// <param name="srcW">Source image width.</param>
    /// <param name="srcH">Source image height.</param>
    /// <param name="shortSide">Target size of the shorter side, typically 800 for Faster R-CNN.</param>
    /// <param name="alignment">Alignment multiple, typically 32.</param>
    /// <param name="mean">Per-channel BGR mean, for example [102.9801f, 115.9465f, 122.7717f].</param>
    /// <returns>CHW BGR float data in a buffer of size 3 * paddedH * paddedW.</returns>
    public static float[] ResizePadBgrNormalize(
        byte[] srcRgb, int srcW, int srcH,
        int shortSide, int alignment,
        float[] mean)
    {
        // Resize so the shorter side matches shortSide.
        float scale = (float)shortSide / Math.Min(srcW, srcH);
        int newW = Math.Max(1, (int)(srcW * scale));
        int newH = Math.Max(1, (int)(srcH * scale));

        // Pad to the requested alignment using bottom-right padding.
        int paddedW = ((newW + alignment - 1) / alignment) * alignment;
        int paddedH = ((newH + alignment - 1) / alignment) * alignment;

        float[] result = new float[3 * paddedW * paddedH]; // Defaults to zero.
        float invScale = 1f / scale;

        int chPlane = paddedW * paddedH;
        int chB = 0;             // B channel offset.
        int chG = chPlane;       // G channel offset.
        int chR = 2 * chPlane;   // R channel offset.

        // Local clamp helpers.
        int ClampY(int v) => v < 0 ? 0 : (v >= srcH ? srcH - 1 : v);
        int ClampX(int v) => v < 0 ? 0 : (v >= srcW ? srcW - 1 : v);

        for (int dy = 0; dy < newH; dy++)
        {
            float sy = dy * invScale;
            int syInt = (int)sy;
            int syBase = syInt - 1;

            // Four vertical weights.
            float wy0 = Cubic(sy - syBase);
            float wy1 = Cubic(sy - (syBase + 1));
            float wy2 = Cubic(sy - (syBase + 2));
            float wy3 = Cubic(sy - (syBase + 3));

            int sy0 = ClampY(syBase);
            int sy1 = ClampY(syBase + 1);
            int sy2 = ClampY(syBase + 2);
            int sy3 = ClampY(syBase + 3);

            int dstRow = dy * paddedW;
            int row0 = sy0 * srcW * 3;
            int row1 = sy1 * srcW * 3;
            int row2 = sy2 * srcW * 3;
            int row3 = sy3 * srcW * 3;

            for (int dx = 0; dx < newW; dx++)
            {
                float sx = dx * invScale;
                int sxInt = (int)sx;
                int sxBase = sxInt - 1;

                // Four horizontal weights.
                float wx0 = Cubic(sx - sxBase);
                float wx1 = Cubic(sx - (sxBase + 1));
                float wx2 = Cubic(sx - (sxBase + 2));
                float wx3 = Cubic(sx - (sxBase + 3));

                int sx0 = ClampX(sxBase);
                int sx1 = ClampX(sxBase + 1);
                int sx2 = ClampX(sxBase + 2);
                int sx3 = ClampX(sxBase + 3);

                int dstIdx = dstRow + dx;
                int x0_3 = sx0 * 3, x1_3 = sx1 * 3, x2_3 = sx2 * 3, x3_3 = sx3 * 3;

                // Apply 4x4 bicubic sampling independently to each channel.
                for (int c = 0; c < 3; c++)
                {
                    float v00 = srcRgb[row0 + x0_3 + c], v01 = srcRgb[row0 + x1_3 + c];
                    float v02 = srcRgb[row0 + x2_3 + c], v03 = srcRgb[row0 + x3_3 + c];
                    float v10 = srcRgb[row1 + x0_3 + c], v11 = srcRgb[row1 + x1_3 + c];
                    float v12 = srcRgb[row1 + x2_3 + c], v13 = srcRgb[row1 + x3_3 + c];
                    float v20 = srcRgb[row2 + x0_3 + c], v21 = srcRgb[row2 + x1_3 + c];
                    float v22 = srcRgb[row2 + x2_3 + c], v23 = srcRgb[row2 + x3_3 + c];
                    float v30 = srcRgb[row3 + x0_3 + c], v31 = srcRgb[row3 + x1_3 + c];
                    float v32 = srcRgb[row3 + x2_3 + c], v33 = srcRgb[row3 + x3_3 + c];

                    float val =
                        (v00 * wx0 + v01 * wx1 + v02 * wx2 + v03 * wx3) * wy0 +
                        (v10 * wx0 + v11 * wx1 + v12 * wx2 + v13 * wx3) * wy1 +
                        (v20 * wx0 + v21 * wx1 + v22 * wx2 + v23 * wx3) * wy2 +
                        (v30 * wx0 + v31 * wx1 + v32 * wx2 + v33 * wx3) * wy3;

                    // BGR channel mapping:
                    // source c=0 (R) -> chR, c=1 (G) -> chG, c=2 (B) -> chB
                    // mean mapping:
                    // c=0 -> mean[2] (R mean), c=1 -> mean[1] (G mean), c=2 -> mean[0] (B mean)
                    int bgrCh = c == 0 ? chR : (c == 1 ? chG : chB);
                    int meanIdx = c == 0 ? 2 : (c == 1 ? 1 : 0);
                    result[bgrCh + dstIdx] = val - mean[meanIdx];
                }
            }
        }

        return result;
    }
}
