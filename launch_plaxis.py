import subprocess
from pathlib import Path

plaxis_path = Path(r"C:/Program Files/Bentley/Geotechnical/PLAXIS 2D CONNECT Edition V21/Plaxis2DXInput.exe")

password = 'admin123'
port = 10000


def main():
    # Launch Plaxis.
    subprocess.Popen([
        plaxis_path.__str__(),
        f'--AppServerPassword={password}',
        f'--AppServerPort={port}'],
        shell=False)

    pass


if __name__ == '__main__':
    main()
