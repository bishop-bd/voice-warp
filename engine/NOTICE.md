# RVC engine dependencies

Voice Warp's worker and process integration are MIT-licensed with the project. This does not relicense its dependencies, models, or shared inference weights.

The runtime downloads the inference code from [w-okada/voice-changer](https://github.com/w-okada/voice-changer/tree/d8ef15799470193f7c8176ef471245753a656626), pinned to commit `d8ef15799470193f7c8176ef471245753a656626`. The upstream copyright notices are retained at `/opt/rvc/LICENSE` in the image and copied to [UPSTREAM-LICENSE](UPSTREAM-LICENSE). The full pinned source remains in the image. Voice Warp imports the RVC and streaming inference classes; it does not launch the upstream HTTP server, web UI, sample downloader, or microphone capture code.

Shared weights downloaded during explicit engine setup:

- [HuBERT base](https://huggingface.co/ddPn08/rvc-webui-models/tree/main/embeddings), SHA-256 `f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96`.
- [RMVPE ONNX](https://huggingface.co/wok000/weights_gpl/tree/main/rmvpe), SHA-256 `84f0586308e36157f75b77c8591bf636d6719c0c4ba95f8faf3df479e7566219`. This comes from the upstream GPL weight collection; its terms are separate from Voice Warp's MIT license.

PyTorch, Fairseq, ONNX Runtime, FAISS, RMVPE, and other dependencies retain their respective licenses and notices. The installer downloads them from their upstream package sources. Distribution of a prebuilt runtime image must account for those licenses and weight terms; the desktop release embeds the recipe and worker rather than redistributing a prebuilt model image. Voice models are supplied separately by the user.

Native runtime candidates built by `scripts/build-ai-runtime.py` contain a frozen worker, native libraries, shared weights, the pinned upstream server source, installed distribution metadata/licenses, and the resolved dependency inventory. These candidates are separate from the desktop package. Their builder does not publish them; redistribution and any corresponding-source obligations must be checked before adding an artifact to the trusted release catalog. CUDA, ROCm, DirectML, and Metal runtime components retain their own vendor terms.
