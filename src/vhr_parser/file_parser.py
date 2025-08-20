import os
from pathlib import Path

from tqdm import tqdm
import polars as pl
import gzip
import struct
import mmap
from concurrent.futures import ProcessPoolExecutor, as_completed


class FileParser:
    """处理二进制文件的解析器，多层同步生成器解析和多进程处理。"""

    def __init__(self, input_dir, output_dir, dbc_parser, max_workers=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        # dbc_parser 是一个外部传入的解析器实例
        self.dbc_parser = dbc_parser
        os.makedirs(output_dir, exist_ok=True)
        self.max_workers = max_workers or os.cpu_count() or 4

    # ---------------------------
    # mmap 读取文件
    # ---------------------------
    @staticmethod
    def mmap_file(file_path):
        """使用 mmap 读取文件，返回 mmap 对象"""

        with file_path.open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # 返回 mmap 对象，可以切片访问
            return mm

    # ---------------------------
    # 四层同步生成器解析
    # ---------------------------
    @staticmethod
    def parse_layer1(mm_view):
        offset = 0
        total_length = len(mm_view)

        while offset + 35 <= total_length:
            header = mm_view[offset : offset + 35]

            if len(header) < 35:
                break

            # 获取 VID
            vid = header[:18].encode("ascii")

            (data_len,) = struct.unpack(">I", header[31:35])

            comp_data = mm_view[offset + 35 : offset + 35 + data_len]

            # 转换为 memoryview
            decompressed_data = memoryview(gzip.decompress(comp_data.tobytes()))

            yield vid, decompressed_data

            offset += 35 + data_len

    @staticmethod
    def parse_layer2(decompressed_data):

        offset = 0

        while offset + 16 <= len(decompressed_data):
            header = decompressed_data[offset : offset + 16]

            # 如果头部不完整，退出循环
            if len(header) < 16:
                break

            (data_len,) = struct.unpack(">H", header[14:16])

            frame_seqs = decompressed_data[offset + 16 : offset + 16 + data_len]

            yield header, frame_seqs

            offset += 16 + data_len

    @staticmethod
    def parse_layer3(frame_seqs):

        offset = 0

        while offset + 8 <= len(frame_seqs):
            header = frame_seqs[offset : offset + 8]

            if len(header) < 8:
                break

            (data_len,) = struct.unpack(">I", header[4:8])

            frame_seq = frame_seqs[offset + 8 : offset + 8 + data_len]

            yield header, frame_seq

            offset += 8 + data_len

    @staticmethod
    def parse_layer4(frame_seq):
        offset = 0

        while offset + 4 <= len(frame_seq):
            header = frame_seq[offset : offset + 4]

            if len(header) < 4:
                break

            (data_len,) = struct.unpack(">H", header[2:4])

            frame = frame_seq[offset + 4 : offset + 4 + data_len]

            yield header, frame

            offset += 4 + data_len

    @staticmethod
    def extract_frames(mm_view):
        for vid, decompressed_data in FileParser.parse_layer1(mm_view):
            for frame_seqs in FileParser.parse_layer2(decompressed_data):
                for frame_seq in FileParser.parse_layer3(frame_seqs):
                    for frame in FileParser.parse_layer3(frame_seq):
                        yield vid, frame

    # ---------------------------
    # 单个文件处理（子进程调用）
    # ---------------------------
    def process_file(self, file_path):
        # mmap 对象
        mm = self.mmap_file(file_path)
        mm_view = memoryview(mm)

        all_rows = []
        for vid, frame in self.extract_frames(mm_view):
            # 使用 dbc_parser 解析帧
            rows = self.dbc_parser.decode_frame(vid, frame)
            # 收集所有解析结果
            all_rows.extend(rows)

        # 获取相对 input 的路径（多层目录保持不变）
        relative_path = file_path.relative_to(self.input_dir)

        # 输出路径 = output + 相对路径
        out_file = self.output_dir / relative_path

        # 确保输出目录存在
        out_file.parent.mkdir(parents=True, exist_ok=True)

        print("输入文件:", file_path)
        print("对应输出文件:", out_file)

        # 输出为 parquet 格式
        out_file = out_file.with_suffix(".parquet")

        df = pl.DataFrame(all_rows)

        # 写入处理后的数据
        df.write_parquet(out_file, compression="snappy")

        # 释放mmap memory视图
        mm_view.release()
        # 关闭文件
        mm.close()

        return out_file

    # ---------------------------
    # 多进程处理目录文件
    # ---------------------------
    def process_directory(self):
        files = [file for file in self.input_dir.rglob("*.bin") if file.is_file()]
        if not files:
            print("没有找到.bin文件")

        pbar = tqdm(total=len(files), desc="任务进度: ")
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.process_file, f): f for f in files}

            try:
                for future in as_completed(future_to_file):
                    f = future_to_file[future]

                    try:
                        out_file = future.result()
                        # 更新tqdm 进度条
                        pbar.update(1)
                        print(f"{f} 解析完成，结果保存到 {out_file}")
                    except Exception as e:
                        print(f"{f} 解析失败: {e}")
            finally:
                pbar.close()


# ---------------------------
# 使用示例
# ---------------------------
if __name__ == "__main__":
    dbc_parser = None  # 假设你有一个 dbc_parser 实例
    parser = FileParser("/path/to/your/files", "/path/to/output/files", dbc_parser)
    parser.process_directory()
