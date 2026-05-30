// Copyright (c) SeasonRealms and contributors.
// Licensed under the Apache License, Version 2.0.
// SeasonOCR for EasyOCR Models

namespace SeasonOCR;

/// <summary>
/// Represents a single recognized text region.
/// </summary>
public class OcrBox
{
    /// <summary>
    /// Gets or sets the left coordinate of the bounding rectangle in source-image pixels.
    /// </summary>
    public int X { get; set; }

    /// <summary>
    /// Gets or sets the top coordinate of the bounding rectangle in source-image pixels.
    /// </summary>
    public int Y { get; set; }

    /// <summary>
    /// Gets or sets the width of the bounding rectangle in source-image pixels.
    /// </summary>
    public int Width { get; set; }

    /// <summary>
    /// Gets or sets the height of the bounding rectangle in source-image pixels.
    /// </summary>
    public int Height { get; set; }

    /// <summary>
    /// Gets or sets the detected quadrilateral in source-image pixels.
    /// For axis-aligned regions this still contains four points.
    /// </summary>
    public Vector2[] Quad { get; set; } = [];

    /// <summary>
    /// Gets or sets the recognized text for the region.
    /// </summary>
    public string Text { get; set; } = "";

    /// <summary>
    /// Gets or sets the confidence score for the recognized text.
    /// </summary>
    public float Confidence { get; set; }

    internal int DebugWinningAngle { get; set; } = -1;

    internal string DebugRotationPolicy { get; set; } = "";
}

/// <summary>
/// Represents a grouped paragraph-level OCR result.
/// </summary>
public class OcrParagraph
{
    /// <summary>
    /// Gets or sets the left coordinate of the paragraph bounding rectangle in source-image pixels.
    /// </summary>
    public int X { get; set; }

    /// <summary>
    /// Gets or sets the top coordinate of the paragraph bounding rectangle in source-image pixels.
    /// </summary>
    public int Y { get; set; }

    /// <summary>
    /// Gets or sets the width of the paragraph bounding rectangle in source-image pixels.
    /// </summary>
    public int Width { get; set; }

    /// <summary>
    /// Gets or sets the height of the paragraph bounding rectangle in source-image pixels.
    /// </summary>
    public int Height { get; set; }

    /// <summary>
    /// Gets or sets the paragraph quadrilateral in source-image pixels.
    /// </summary>
    public Vector2[] Quad { get; set; } = [];

    /// <summary>
    /// Gets or sets the aggregated paragraph text.
    /// </summary>
    public string Text { get; set; } = "";

    /// <summary>
    /// Gets or sets the aggregated paragraph confidence score.
    /// </summary>
    public float Confidence { get; set; }
}

/// <summary>
/// Represents a single debug entry for a detected OCR region before paragraph aggregation.
/// </summary>
public sealed class OcrDebugBox
{
    /// <summary>
    /// Gets or sets the zero-based index of the detected region.
    /// </summary>
    public int Index { get; set; }

    /// <summary>
    /// Gets or sets the left coordinate of the detected region in source-image pixels.
    /// </summary>
    public int X { get; set; }

    /// <summary>
    /// Gets or sets the top coordinate of the detected region in source-image pixels.
    /// </summary>
    public int Y { get; set; }

    /// <summary>
    /// Gets or sets the width of the detected region in source-image pixels.
    /// </summary>
    public int Width { get; set; }

    /// <summary>
    /// Gets or sets the height of the detected region in source-image pixels.
    /// </summary>
    public int Height { get; set; }

    /// <summary>
    /// Gets or sets the detected quadrilateral in source-image pixels.
    /// </summary>
    public Vector2[] Quad { get; set; } = [];

    /// <summary>
    /// Gets or sets the recognized text for the detected region.
    /// </summary>
    public string Text { get; set; } = "";

    /// <summary>
    /// Gets or sets the confidence score for the recognized text.
    /// </summary>
    public float Confidence { get; set; }

    /// <summary>
    /// Gets or sets the winning recognition rotation angle in degrees.
    /// </summary>
    public int? WinningAngle { get; set; }

    /// <summary>
    /// Gets or sets the rotation policy that was applied to the region.
    /// </summary>
    public string RotationPolicy { get; set; } = "";

    /// <summary>
    /// Gets or sets whether the region is kept in the final non-empty OCR output.
    /// </summary>
    public bool IncludedInResult { get; set; }
}

/// <summary>
/// Represents the full OCR result for a processed image.
/// </summary>
public class SeasonOcrResult
{
    /// <summary>
    /// Gets or sets the human-readable summary produced from the recognized text.
    /// </summary>
    public string Summary { get; set; } = "";

    /// <summary>
    /// Gets or sets the annotated output image encoded as JPEG bytes.
    /// </summary>
    public byte[] AnnotatedImage { get; set; } = [];

    /// <summary>
    /// Gets or sets the individual recognized text boxes.
    /// </summary>
    public List<OcrBox> Boxes { get; set; } = [];

    /// <summary>
    /// Gets or sets the paragraph-level grouping generated from the recognized boxes.
    /// </summary>
    public List<OcrParagraph> Paragraphs { get; set; } = [];

    /// <summary>
    /// Gets or sets the per-box debug entries generated before paragraph aggregation.
    /// </summary>
    public List<OcrDebugBox> DebugBoxes { get; set; } = [];

    /// <summary>
    /// Gets or sets a human-readable debug report that can be saved to a UTF-8 text file.
    /// </summary>
    public string DebugReport { get; set; } = "";
}
