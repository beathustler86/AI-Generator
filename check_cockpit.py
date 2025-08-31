# ✅ Core AI stack with CUDA 12.4
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0

# 🧠 Diffusion, Transformers, ONNX
diffusers==0.35.1
transformers==4.53.3
onnx==1.18.0
onnxruntime==1.22.1
optimum==1.27.0
torchsde==0.2.6
sentencepiece==0.2.1
tokenizers==0.21.4

# 🖼️ Image & Audio Processing
opencv-python==4.12.0.88
scikit-image==0.25.2
imageio==2.37.0
pillow==11.3.0
soundfile==0.13.1
pydub==0.25.1
tifffile==2025.5.10
av==15.0.0
ffmpy==0.6.1

# 🎯 Enhancement & Restoration
realesrgan @ git+https://github.com/xinntao/Real-ESRGAN.git@a4abfb2979a7bbff3f69f58f58ae324608821e27
facexlib==0.3.0
gfpgan==1.3.8
basicsr==1.4.2

# ⚙️ FastAPI & Gradio UI
fastapi==0.116.1
uvicorn==0.35.0
gradio==5.44.0
gradio_client==1.12.1
starlette==0.47.3
Werkzeug==3.1.3

# 🧰 Utilities & Logging
numpy>=1.22,<2.3.0
scipy==1.15.3
psutil==7.0.0
coloredlogs==15.0.1
rich==14.1.0
tqdm==4.67.1
attrs==25.3.0
python-dotenv==1.1.1
humanfriendly==10.0
GPUtil==1.4.0
filelock==3.19.1
safehttpx==0.1.6
safetensors==0.6.2

# 🛡️ Validation & Schema
pydantic==2.11.7
pydantic-settings==2.10.1
pydantic_core==2.33.2
jsonschema==4.25.1
jsonschema-specifications==2025.4.1
referencing==0.36.2
fastjsonschema==2.21.2

# 🧪 Dev Tools & Jupyter
ipython==8.12.3
ipykernel==6.30.1
jupyter_client==8.6.3
jupyter_core==5.8.1
matplotlib==3.10.5
matplotlib-inline==0.1.7
nbclient==0.10.2
nbconvert==7.16.6
nbformat==5.10.4
jupyterlab_pygments==0.3.0

# 🧼 Git Hygiene & Packaging
pipreqs==0.5.0
ruff==0.12.11
yapf==0.43.0
vulture==2.14
semantic-version==2.10.0
pyinstaller==6.15.0
pyinstaller-hooks-contrib==2025.8

# 📦 Data & Serialization
protobuf==6.32.0
orjson==3.11.3
pyarrow==21.0.0
PyYAML==6.0.2
lmdb==1.7.3

# 🔗 Networking & HTTP
aiohttp==3.12.15
aiofiles==24.1.0
aiohappyeyeballs==2.6.1
aiosignal==1.4.0
httpx==0.28.1
httpcore==1.0.9
websockets==15.0.1
urllib3==2.5.0
requests==2.32.5
idna==3.10
charset-normalizer==3.4.3

# 🧠 ML & Data Science
accelerate==0.23.0
datasets==3.6.0
numba==0.61.2
networkx==3.4.2
filterpy==1.4.5
multiprocess==0.70.16
pandas==2.3.2

# 🧩 ComfyUI & Extensions
comfy==0.0.1
comfyui-embedded-docs==0.2.6
comfyui_frontend_package==1.25.11
comfyui_workflow_templates==0.1.68
