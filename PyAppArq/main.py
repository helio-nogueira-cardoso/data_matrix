#!/usr/bin/env python3
"""
PyAppArq - Reescrita em Python do AppArq para analise de maquetes arquitetonicas.

Substitui a aplicacao C++/Qt por uma implementacao Python/GTK3.
Usa decodificacao DataMatrix ECC200 (via pylibdmtx) em vez de SIFT.
Produz o mesmo formato JSON de saida para integracao com o Revit.

Uso:
    python main.py
"""

from gui import App


if __name__ == "__main__":
    app = App()
    app.run()
