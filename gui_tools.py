# gui_tools.py
# Backward-compatible entrypoint exporting tools from their respective modular files.

from gui_tool_base import BaseTool
from gui_tool_trim import TrimTool
from gui_tool_preprocess import PreprocessTool
from gui_tool_process import ProcessVideoTool
from gui_tool_visualize import VisualizeVideoTool

EXPOSED_TOOLS = [
    TrimTool,
    PreprocessTool,
    ProcessVideoTool,
    VisualizeVideoTool
]