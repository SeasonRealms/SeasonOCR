// Copyright (c) SeasonRealms and contributors.
// Licensed under the Apache License, Version 2.0.
// SeasonOCR for EasyOCR Models

using StbImageWriteSharp;

namespace SeasonOCR;

internal static class ImageCodec
{
    public static byte[] SaveAsJpeg(ImageResult image, int quality = 90)
    {
        byte[] rgb;
        int pixelCount = image.Width * image.Height;

        if (image.Data.Length == pixelCount * 3)
        {
            rgb = image.Data;
        }
        else if (image.Data.Length == pixelCount * 4)
        {
            rgb = new byte[pixelCount * 3];
            for (int i = 0; i < pixelCount; i++)
            {
                int src = i * 4;
                int dst = i * 3;
                rgb[dst] = image.Data[src];
                rgb[dst + 1] = image.Data[src + 1];
                rgb[dst + 2] = image.Data[src + 2];
            }
        }
        else
        {
            throw new NotSupportedException(
                $"SaveAsJpeg does not support buffer length {image.Data.Length}. " +
                $"Pixel count is {pixelCount}; expected {pixelCount * 3} (RGB) or {pixelCount * 4} (RGBA).");
        }

        using var output = new MemoryStream();
        var encoder = new ImageWriter();
        encoder.WriteJpg(rgb, image.Width, image.Height, StbImageWriteSharp.ColorComponents.RedGreenBlue, output, quality);
        return output.ToArray();
    }
}
