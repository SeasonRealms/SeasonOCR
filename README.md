# SeasonOCR

SeasonOCR for EasyOCR Models is a cross-platform .NET OCR library that runs ONNX models converted from the EasyOCR Python project.

## Origin

- The OCR pipeline is translated and adapted from [EasyOCR](https://github.com/JaidedAI/EasyOCR).
- The intended model inputs are ONNX files converted from EasyOCR Python models.
- Files translated from EasyOCR Python sources include source attribution and an explicit modification notice in the file header.

## Features

- EasyOCR-style CRAFT post-processing
- CRNN / CTC recognition
- Beam search by default, with optional dictionary-aware word beam search
- Perspective rectification for rotated text regions
- Optional rotation TTA with best-confidence selection, disabled by default
- Paragraph grouping with structured OCR output
- Dictionary loading compatible with EasyOCR model layouts
- Automatic recognizer language inference from ONNX metadata, with charset and filename fallback

## Install

```bash
dotnet add package SeasonOCR
```

## Usage

```csharp
using SeasonOCR;
using StbImageSharp;

using var stream = File.OpenRead("input.png");
var image = ImageResult.FromStream(stream, ColorComponents.RedGreenBlue);

var result = SeasonOcr.DetectDetailed(
    detectorModelPath: "craft_mlt_25k.onnx",
    recognizerModelPath: "recognizer_en.onnx",
    imageResult: image,
    enableWordBeamSearch: true,
    allowRotatedRecognition: false,
    beamWidth: 5);

Console.WriteLine(result.Summary);
foreach (var paragraph in result.Paragraphs)
{
    Console.WriteLine($"{paragraph.Confidence:P0}: {paragraph.Text}");
}

File.WriteAllBytes("annotated.jpg", result.AnnotatedImage);
```

- The annotated output image draws detection boxes only, matching the official EasyOCR-style preview behavior.

## Models And Dictionaries

- `detectorModelPath` and `recognizerModelPath` accept local file paths.
- `SeasonOcr` reads recognizer `lang_list` metadata automatically when it is embedded in the ONNX model.
- If language metadata is missing, the library falls back to charset heuristics and then filename heuristics.
- `BeamSearch` is the default decoder for all languages.
- `allowRotatedRecognition` is disabled by default to reduce false positives on small or ambiguous boxes.
- `enableWordBeamSearch` only applies dictionary constraints to segments that can be matched to an available language dictionary; other segments continue to use plain beam search.
- Word beam search prefers an explicit `dictionaryPath` when one is provided.
- If no explicit dictionary is passed, the library probes the recognizer directory, a `dict/` subdirectory, and an `EasyOCR-1.7.2/easyocr/dict/` layout near the running app.

## License

- The project is distributed under Apache License 2.0.
- See `LICENSE` and `THIRD_PARTY_NOTICES.md` for attribution and third-party details.
