# ECC200Decode

Pipeline para ortorretificação, segmentação em grade (37x37) e
decodificação de códigos DataMatrix ECC200.

## Requisitos

### Sistema (Debian/Ubuntu)

sudo apt update sudo apt install -y python3 python3-venv python3-pip
libdmtx0t64 libdmtx-dev

### Python

Python 3.11+

## Instalação

git clone `<repo>`{=html} cd ECC200Decode

python3 -m venv venv source venv/bin/activate

pip install --upgrade pip pip install -r requirements.txt

## requirements.txt

pylibdmtx Pillow opencv-python numpy

## Uso

python pipeline.py

## Uso com parâmetros

python pipeline.py --input imagem.png --template template.png --output
ortho.png --grid-output grid.txt

## Debug

python pipeline.py --test-cell 5,6 python pipeline.py --dump-elements

## Problemas comuns

### libdmtx

sudo apt install libdmtx0t64 libdmtx-dev

### cv2

pip install opencv-python

## Saída

Arquivo grid.txt com matriz 37x37

## Autor

Hélio Nogueira Cardoso
