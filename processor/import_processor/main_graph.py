"""知识库导入工作流。"""

import argparse
import json
import sys
from pathlib import Path

# 支持直接执行本文件；使用 -m 时不需要这段路径引导。
if __package__ in (None, ""):
    project_root = str(Path(__file__).resolve().parents[2])
    # PyCharm可能已经把项目根目录放在sys.path后面。
    # 必须先移除再放到最前面，否则同目录的config.py会抢占根目录config包。
    while project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.node_md_img import NodeMDImg
from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_processor.state import ImportGraphState, get_default_state


class KBImportWorkflow:
    """PDF/Markdown 导入知识库的完整工作流。"""

    def __init__(self):
        self.__compiled_graph = None

    @property
    def graph(self):
        if self.__compiled_graph is None:
            self.__compiled_graph = self.build_graph()
        return self.__compiled_graph

    @staticmethod
    def route_after_entry(state: ImportGraphState):
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        if state.get("is_md_read_enabled"):
            return "node_md_img"
        return END

    def build_graph(self):
        graph = StateGraph(ImportGraphState)
        graph.add_node("node_entry", NodeEntry())
        graph.add_node("node_pdf_to_md", NodePDFToMD())
        graph.add_node("node_md_img", NodeMDImg())
        graph.add_node("node_document_split", NodeDocumentSplit())
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("node_import_milvus", NodeImportMilvus())

        graph.set_entry_point("node_entry")
        graph.add_conditional_edges(
            "node_entry",
            self.route_after_entry,
            {
                "node_pdf_to_md": "node_pdf_to_md",
                "node_md_img": "node_md_img",
                END: END,
            },
        )
        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)
        return graph.compile()

    def run(self, state: ImportGraphState, stream: bool = False):
        if stream:
            return self.graph.stream(state, stream_mode="values")
        return self.graph.invoke(state)


def _build_initial_state(import_file_path: str) -> ImportGraphState:
    state = get_default_state()
    state["import_file_path"] = import_file_path
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="导入 PDF 或 Markdown 到知识库")
    parser.add_argument(
        "import_file_path",
        nargs="?",
        help="需要导入的 PDF/Markdown 文件路径；不传时只检查工作流能否构图",
    )
    parser.add_argument("--print-graph", action="store_true", help="打印 LangGraph 结构")
    args = parser.parse_args()

    setup_logging()
    workflow = KBImportWorkflow()
    compiled_graph = workflow.graph
    print("导入工作流构图成功")

    if args.print_graph:
        compiled_graph.get_graph().print_ascii()

    if not args.import_file_path:
        return 0

    final_state = workflow.run(
        _build_initial_state(args.import_file_path),
        stream=False,
    )
    summary = {
        "import_file_path": final_state.get("import_file_path"),
        "md_path": final_state.get("md_path"),
        "file_title": final_state.get("file_title"),
        "item_name": final_state.get("item_name"),
        "chunk_count": len(final_state.get("chunks") or []),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
