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
    private static float Sigmoid(float x)
    {
        if (x >= 0)
            return 1.0f / (1.0f + MathF.Exp(-x));
        float expX = MathF.Exp(x);
        return expX / (1.0f + expX);
    }
}
