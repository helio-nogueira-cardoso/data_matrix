"""
gui.py - Interface grafica GTK3 para o PyAppArq.

Fornece:
- Janela principal: selecao de modo (Interativo / Automatico), carregamento
  de imagem, processamento.
- Janela de correcao: mostra a imagem ortorretificada com simbolos
  decodificados sobrepostos. O usuario pode clicar nas celulas para
  corrigir simbolos mal identificados.
- Funcionalidade de salvamento: interpretacao semantica + exportacao
  JSON para o Revit.
"""

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib, Pango

from objects_handler import ObjectsHandler
from pipeline import COLS, ROWS, process_image


@dataclass
class GridCell:
    """Estado de exibicao de uma celula da grade."""
    symbol: str = "-"
    original_symbol: str = "-"  # valor decodificado, usado para detectar correcoes reais
    is_uncertain: bool = False
    is_changed: bool = False


# ======================================================================
# Janela Principal da Aplicacao
# ======================================================================

class App:
    """Janela principal da aplicacao."""

    def __init__(self):
        self._image_path = None
        self._script_dir = Path(__file__).resolve().parent
        self._template_path = str(self._script_dir / "template.png")
        self._objects_path = str(self._script_dir / "objetos.json")

        self.window = Gtk.Window(title="PyAppArq")
        self.window.set_default_size(480, 360)
        self.window.set_position(Gtk.WindowPosition.CENTER)
        self.window.set_resizable(False)
        self.window.connect("destroy", Gtk.main_quit)

        self._build_ui()

    def _build_ui(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        vbox.set_margin_top(20)
        vbox.set_margin_bottom(20)
        vbox.set_margin_start(24)
        vbox.set_margin_end(24)
        self.window.add(vbox)

        # Titulo
        title = Gtk.Label(label="PyAppArq")
        title.set_markup("<span size='xx-large' weight='bold'>PyAppArq</span>")
        vbox.pack_start(title, False, False, 0)

        # Quadro de selecao de modo
        mode_frame = Gtk.Frame(label=" Modo de operacao ")
        mode_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        mode_box.set_margin_top(8)
        mode_box.set_margin_bottom(8)
        mode_box.set_margin_start(12)
        mode_box.set_margin_end(12)
        mode_frame.add(mode_box)
        vbox.pack_start(mode_frame, False, False, 0)

        self._radio_interactive = Gtk.RadioButton.new_with_label_from_widget(
            None, "Interativo (verificar simbolos)")
        self._radio_automatic = Gtk.RadioButton.new_with_label_from_widget(
            self._radio_interactive, "Automatico (aceitar tudo)")
        mode_box.pack_start(self._radio_interactive, False, False, 0)
        mode_box.pack_start(self._radio_automatic, False, False, 0)

        # Quadro de selecao de imagem
        img_frame = Gtk.Frame(label=" Imagem de entrada ")
        img_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        img_box.set_margin_top(8)
        img_box.set_margin_bottom(8)
        img_box.set_margin_start(12)
        img_box.set_margin_end(12)
        img_frame.add(img_box)
        vbox.pack_start(img_frame, False, False, 0)

        btn_select = Gtk.Button(label="Selecionar Imagem")
        btn_select.connect("clicked", self._on_select_image)
        img_box.pack_start(btn_select, False, False, 0)

        self._path_label = Gtk.Label(label="Nenhuma imagem selecionada")
        self._path_label.set_xalign(0)
        self._path_label.set_line_wrap(True)
        self._path_label.set_opacity(0.6)
        img_box.pack_start(self._path_label, False, False, 0)

        # Botao de processar
        self._process_btn = Gtk.Button(label="Processar")
        self._process_btn.connect("clicked", self._on_process)
        vbox.pack_start(self._process_btn, False, False, 4)

        # Label de status
        self._status_label = Gtk.Label(label="")
        self._status_label.set_xalign(0.5)
        vbox.pack_start(self._status_label, False, False, 0)

    def _on_select_image(self, _widget):
        dialog = Gtk.FileChooserDialog(
            title="Selecionar imagem da maquete",
            parent=self.window,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK,
        )
        filt = Gtk.FileFilter()
        filt.set_name("Imagens")
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            filt.add_pattern(pattern)
        dialog.add_filter(filt)

        filt_all = Gtk.FileFilter()
        filt_all.set_name("Todos")
        filt_all.add_pattern("*")
        dialog.add_filter(filt_all)

        if dialog.run() == Gtk.ResponseType.OK:
            self._image_path = dialog.get_filename()
            self._path_label.set_text(os.path.basename(self._image_path))
        dialog.destroy()

    def _on_process(self, _widget):
        if not self._image_path:
            _show_message(self.window, "Aviso", "Selecione uma imagem primeiro.",
                          Gtk.MessageType.WARNING)
            return

        self._process_btn.set_sensitive(False)
        self._status_label.set_text("Processando...")

        def run():
            try:
                ortho, grid, boxes = process_image(
                    self._image_path,
                    self._template_path,
                    progress_callback=lambda msg: GLib.idle_add(
                        self._status_label.set_text, msg
                    ),
                )
                GLib.idle_add(self._on_done, ortho, grid, boxes)
            except Exception as e:
                GLib.idle_add(self._on_error, str(e))

        threading.Thread(target=run, daemon=True).start()

    def _on_done(self, ortho, grid, boxes):
        self._process_btn.set_sensitive(True)

        # Constroi a matriz de GridCell.
        cells = [[GridCell() for _ in range(COLS)] for _ in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                sym = grid[r][c]
                if sym == "?":
                    cells[r][c] = GridCell(symbol="?", original_symbol="?",
                                          is_uncertain=True)
                else:
                    cells[r][c] = GridCell(symbol=sym, original_symbol=sym)

        interactive = self._radio_interactive.get_active()

        if interactive:
            win = CorrectionWindow(self.window, ortho, cells, boxes,
                                   self._image_path, self._objects_path)
            win.show()
            self.window.hide()
        else:
            # Modo automatico: vai direto para o salvamento.
            self._auto_save(ortho, cells)

    def _on_error(self, msg):
        self._process_btn.set_sensitive(True)
        self._status_label.set_text("Erro!")
        _show_message(self.window, "Erro no processamento", msg,
                      Gtk.MessageType.ERROR)

    def _auto_save(self, ortho, cells):
        """Modo automatico: pede local de saida e salva diretamente."""
        project_name = _ask_string(self.window, "Nome do Projeto",
                                   "Informe o nome do projeto:")
        if not project_name:
            self._status_label.set_text("Cancelado.")
            return

        out_dir = _ask_directory(self.window, "Selecionar diretorio de saida")
        if not out_dir:
            self._status_label.set_text("Cancelado.")
            return

        grid = [[c.symbol for c in row] for row in cells]
        _save_project(project_name, out_dir, grid, ortho,
                      self._image_path, self._objects_path)
        self._status_label.set_text(f"Projeto '{project_name}' salvo com sucesso.")

    def run(self):
        self.window.show_all()
        Gtk.main()


# ======================================================================
# Janela de Correcao
# ======================================================================

class CorrectionWindow:
    """Janela interativa para verificacao e correcao de simbolos."""

    # Esquema de cores igual ao AppArq: vermelho=incerto, azul=confiante, verde=corrigido
    COLOR_UNCERTAIN = (0.8, 0.0, 0.0)
    COLOR_NORMAL = (0.0, 0.0, 0.8)
    COLOR_CHANGED = (0.0, 0.67, 0.0)

    def __init__(self, parent, ortho_bgr, cells, boxes, image_path, objects_path):
        self.parent = parent
        self.ortho_bgr = ortho_bgr
        self.cells = cells
        self.boxes = boxes
        self.image_path = image_path
        self.objects_path = objects_path

        # Calcula centros das celulas em coordenadas da ortho.
        self.centers = {}
        for r, c, x0, y0, x1, y1 in boxes:
            self.centers[(r, c)] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

        # Calcula tamanho da celula para dimensionar a fonte.
        if boxes:
            _, _, x0, y0, x1, y1 = boxes[0]
            self.cell_side = x1 - x0
        else:
            self.cell_side = 20

        # Calcula escala para caber na tela.
        screen = Gdk.Screen.get_default()
        screen_w = screen.get_width()
        screen_h = screen.get_height()
        img_h, img_w = ortho_bgr.shape[:2]

        max_w = int(screen_w * 0.80)
        max_h = int(screen_h * 0.80)
        self.scale = min(max_w / img_w, max_h / img_h, 1.0)
        self.disp_w = int(img_w * self.scale)
        self.disp_h = int(img_h * self.scale)

        # Prepara pixbuf escalado.
        self._prepare_pixbuf()

        # Constroi a janela.
        self.window = Gtk.Window(title="Verificacao de Simbolos")
        self.window.set_default_size(
            min(self.disp_w + 40, screen_w - 40),
            min(self.disp_h + 100, screen_h - 40),
        )
        self.window.set_position(Gtk.WindowPosition.CENTER)
        self.window.connect("delete-event", self._on_close)

        self._build_ui()

    def _prepare_pixbuf(self):
        """Converte a ortho BGR para GdkPixbuf na escala de exibicao."""
        rgb = cv2.cvtColor(self.ortho_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * ch,
        )
        self._pixbuf = pixbuf.scale_simple(
            self.disp_w, self.disp_h, GdkPixbuf.InterpType.BILINEAR,
        )

    def _build_ui(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.window.add(vbox)

        # Barra de ferramentas
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_margin_top(6)
        toolbar.set_margin_bottom(4)
        toolbar.set_margin_start(8)
        toolbar.set_margin_end(8)
        vbox.pack_start(toolbar, False, False, 0)

        hint = Gtk.Label(label="Clique em um simbolo para corrigir.")
        hint.set_opacity(0.6)
        toolbar.pack_start(hint, False, False, 0)

        btn_save = Gtk.Button(label="Salvar Projeto")
        btn_save.connect("clicked", self._on_save)
        toolbar.pack_end(btn_save, False, False, 0)

        btn_back = Gtk.Button(label="Voltar")
        btn_back.connect("clicked", self._on_close)
        toolbar.pack_end(btn_back, False, False, 0)

        # Barra de status
        self._status_label = Gtk.Label(label="")
        self._status_label.set_margin_start(8)
        self._status_label.set_xalign(0)
        vbox.pack_start(self._status_label, False, False, 0)

        # Area de desenho com rolagem
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        vbox.pack_start(scrolled, True, True, 0)

        self._drawing_area = Gtk.DrawingArea()
        self._drawing_area.set_size_request(self.disp_w, self.disp_h)
        self._drawing_area.connect("draw", self._on_draw)
        self._drawing_area.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self._drawing_area.connect("button-press-event", self._on_click)
        scrolled.add(self._drawing_area)

        self._update_status()

    def _on_draw(self, widget, cr):
        """Renderiza a imagem e sobrepoe os simbolos via Cairo."""
        # Desenha a imagem.
        Gdk.cairo_set_source_pixbuf(cr, self._pixbuf, 0, 0)
        cr.paint()

        # Calcula tamanho da fonte.
        font_size = max(7, min(16, int(self.cell_side * self.scale * 0.45)))
        cr.select_font_face("monospace", 0, 1)  # NORMAL, BOLD
        cr.set_font_size(font_size)

        for r in range(ROWS):
            for c in range(COLS):
                cell = self.cells[r][c]

                # Pula celulas realmente vazias.
                if cell.symbol == "-":
                    continue
                # Mostra "?" para celulas incertas para que o usuario possa corrigir.
                if cell.symbol == "?" and not cell.is_uncertain:
                    continue

                cx, cy = self.centers.get((r, c), (0, 0))
                sx = cx * self.scale
                sy = cy * self.scale

                if cell.is_changed:
                    color = self.COLOR_CHANGED
                elif cell.is_uncertain:
                    color = self.COLOR_UNCERTAIN
                else:
                    color = self.COLOR_NORMAL

                # Desenha texto com sombra leve para legibilidade.
                extents = cr.text_extents(cell.symbol)
                tx = sx - extents.width / 2
                ty = sy + extents.height / 2

                # Sombra
                cr.set_source_rgba(1, 1, 1, 0.7)
                cr.move_to(tx - 1, ty + 1)
                cr.show_text(cell.symbol)

                # Primeiro plano
                cr.set_source_rgb(*color)
                cr.move_to(tx, ty)
                cr.show_text(cell.symbol)

    def _on_click(self, widget, event):
        """Trata o clique: encontra a celula mais proxima e solicita correcao."""
        img_x = event.x / self.scale
        img_y = event.y / self.scale

        # Encontra a celula mais proxima.
        best_r, best_c = 0, 0
        best_dist = float("inf")
        for r in range(ROWS):
            for c in range(COLS):
                cell_cx, cell_cy = self.centers.get((r, c), (0, 0))
                dist = (img_x - cell_cx) ** 2 + (img_y - cell_cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_r, best_c = r, c

        cell = self.cells[best_r][best_c]
        current = cell.symbol if cell.symbol != "-" else ""

        new_val = _ask_string(
            self.window,
            "Correcao de Simbolo",
            f"Celula ({best_r}, {best_c})\n"
            f"Valor atual: '{current}'\n\n"
            f"Novo valor (vazio = marcar como vazio):",
            default=current,
        )

        if new_val is None:
            return  # cancelado

        new_val = new_val.strip()
        if new_val == "":
            new_val = "-"
        cell.symbol = new_val

        # So marca como alterado se diferente do valor original decodificado.
        if new_val == cell.original_symbol:
            cell.is_changed = False
            cell.is_uncertain = (cell.original_symbol == "?")
        else:
            cell.is_changed = True
            cell.is_uncertain = False

        self._drawing_area.queue_draw()
        self._update_status()

    def _update_status(self):
        decoded = sum(1 for row in self.cells for c in row
                      if c.symbol not in ("?", "-"))
        uncertain = sum(1 for row in self.cells for c in row if c.is_uncertain)
        changed = sum(1 for row in self.cells for c in row if c.is_changed)
        self._status_label.set_text(
            f"Decodificados: {decoded}  |  "
            f"Incertos: {uncertain}  |  "
            f"Corrigidos: {changed}"
        )

    def _on_save(self, _widget=None):
        project_name = _ask_string(self.window, "Nome do Projeto",
                                   "Informe o nome do projeto:")
        if not project_name:
            return

        out_dir = _ask_directory(self.window, "Selecionar diretorio de saida")
        if not out_dir:
            return

        grid = [[c.symbol for c in row] for row in self.cells]
        _save_project(project_name, out_dir, grid, self.ortho_bgr,
                      self.image_path, self.objects_path)

        _show_message(self.window, "Sucesso",
                      f"Projeto '{project_name}' salvo com sucesso.",
                      Gtk.MessageType.INFO)
        self._close()

    def _on_close(self, _widget=None, _event=None):
        self._close()
        return True

    def _close(self):
        self.window.destroy()
        self.parent.show()

    def show(self):
        self.window.show_all()


# ======================================================================
# Dialogos auxiliares GTK
# ======================================================================

def _show_message(parent, title, text, msg_type=Gtk.MessageType.INFO):
    """Exibe uma mensagem modal ao usuario."""
    dialog = Gtk.MessageDialog(
        transient_for=parent,
        flags=Gtk.DialogFlags.MODAL,
        message_type=msg_type,
        buttons=Gtk.ButtonsType.OK,
        text=title,
    )
    dialog.format_secondary_text(text)
    dialog.run()
    dialog.destroy()


def _ask_string(parent, title, prompt, default=""):
    """Dialogo de entrada de texto. Retorna string ou None se cancelado."""
    dialog = Gtk.Dialog(
        title=title,
        transient_for=parent,
        flags=Gtk.DialogFlags.MODAL,
    )
    dialog.add_buttons(
        Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
        Gtk.STOCK_OK, Gtk.ResponseType.OK,
    )
    dialog.set_default_response(Gtk.ResponseType.OK)

    box = dialog.get_content_area()
    box.set_margin_top(12)
    box.set_margin_bottom(8)
    box.set_margin_start(12)
    box.set_margin_end(12)
    box.set_spacing(8)

    label = Gtk.Label(label=prompt)
    label.set_xalign(0)
    label.set_line_wrap(True)
    box.add(label)

    entry = Gtk.Entry()
    entry.set_text(default)
    entry.set_activates_default(True)
    box.add(entry)

    dialog.show_all()
    response = dialog.run()
    text = entry.get_text() if response == Gtk.ResponseType.OK else None
    dialog.destroy()
    return text


def _ask_directory(parent, title):
    """Dialogo de selecao de diretorio. Retorna caminho ou None."""
    dialog = Gtk.FileChooserDialog(
        title=title,
        parent=parent,
        action=Gtk.FileChooserAction.SELECT_FOLDER,
    )
    dialog.add_buttons(
        Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
        Gtk.STOCK_OPEN, Gtk.ResponseType.OK,
    )
    result = None
    if dialog.run() == Gtk.ResponseType.OK:
        result = dialog.get_filename()
    dialog.destroy()
    return result


# ======================================================================
# Logica de salvamento
# ======================================================================

def _save_project(project_name, out_dir, grid, ortho_bgr,
                  image_path, objects_path):
    """Executa a interpretacao semantica e salva JSON + imagem."""
    folder = Path(out_dir) / project_name
    folder.mkdir(parents=True, exist_ok=True)

    # Interpretacao semantica
    handler = ObjectsHandler(objects_path)
    handler.find_objects_in_grid(grid)
    objects = handler.format_objects_json()

    # Escreve o JSON
    json_path = folder / "maquete_objetos.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(objects, f, indent=2, ensure_ascii=False)

    # Salva a imagem original
    img_path = folder / "maquete_imagem.png"
    original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original is not None:
        cv2.imwrite(str(img_path), original)

    # Salva tambem a grade como texto para referencia
    grid_path = folder / "maquete_grade.txt"
    grid_text = "\n".join(" ".join(row) for row in grid)
    grid_path.write_text(grid_text, encoding="utf-8")

    print(f"Projeto salvo em: {folder}")
    print(f"  Elementos: {handler.summary()}")
