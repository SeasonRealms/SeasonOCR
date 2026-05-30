# THIRD_PARTY_NOTICES

SeasonOCR includes or adapts logic from the following third-party projects.

## EasyOCR

- Project: https://github.com/JaidedAI/EasyOCR
- Copyright: EasyOCR authors and contributors
- License: Apache License 2.0
- Usage: Detection/recognition flow, decoder behavior, dictionary loading, separator-based word beam search logic, and related OCR pipeline adaptations

## CRAFT / Related Detection Logic

- Source files referenced by EasyOCR include CRAFT-related components with MIT-licensed notices
- License: MIT License
- Usage: Parts of the text detection post-processing and box handling pipeline trace back to this family of implementations through EasyOCR

## Microsoft.ML.OnnxRuntime

- Package: Microsoft.ML.OnnxRuntime
- License: MIT License
- Usage: ONNX model inference runtime

## StbImageSharp

- Package: StbImageSharp
- License: MIT License
- Usage: Image loading and decoded image container

## StbImageWriteSharp

- Package: StbImageWriteSharp
- License: MIT License
- Usage: JPEG encoding for annotated output

## Notes

- Files translated or adapted from EasyOCR Python sources include a source notice and an explicit modification statement in the file header.
- This project name is SeasonOCR; EasyOCR is referenced only to identify model provenance and implementation origin.
