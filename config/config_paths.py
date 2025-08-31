# src/config/config_paths.py


from pathlib import Path

UPSCALE_DIR = Path(r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\Upscaler\upscaler-ultra")

UPSCALE_MODEL_PATHS = dict(sorted({
	"Anime6B": UPSCALE_DIR / "RealESRGAN_x4plus_anime_6B.pth",
	"Remacri": UPSCALE_DIR / "4x_foolhardy_Remacri.pth",
	"UltraSharp": UPSCALE_DIR / "4x-UltraSharp.pth"
}.items()))


