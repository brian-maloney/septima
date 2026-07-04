# Third-party notices

septima is MIT-licensed (see [LICENSE](LICENSE)). `cmd/septima` additionally
embeds and `go build` links the following third-party components; this file
covers attribution for those.

## ONNX Runtime (embedded, vendored under `internal/ortlib/lib/`)

`cmd/septima` embeds a per-platform build of the
[ONNX Runtime](https://github.com/microsoft/onnxruntime) shared library
(darwin/arm64, linux/amd64, linux/arm64, windows/amd64), unmodified, as
downloaded from Microsoft's official releases. ONNX Runtime is MIT-licensed,
Copyright (c) Microsoft Corporation — see `internal/ortlib/LICENSE`.

ONNX Runtime's own binary distribution bundles third-party components (Eigen,
protobuf, NumPy, Caffe2/PyTorch, RE2, nlohmann/json, Intel MKL, and others)
under their own licenses (mostly MIT/BSD/Apache-2.0, plus Eigen under
Mozilla Public License 2.0 and Intel MKL under the Intel Simplified Software
License). None of these are strong/viral copyleft (no GPL/LGPL/AGPL
component), and none impose obligations on septima beyond attribution: we
redistribute the ONNX Runtime binary unmodified, we don't statically link or
modify any of its bundled components ourselves. The full, unmodified notices
file as shipped by Microsoft is reproduced at
`internal/ortlib/THIRD_PARTY_NOTICES.txt`.

## onnxruntime_go (Go dependency, all `cmd/` binaries)

[github.com/yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go),
the cgo binding used to drive ONNX Runtime from Go, is MIT-licensed,
Copyright (c) 2023 Nathan Otterness.
