import os
import hashlib
from datetime import datetime
def get_file_hash(file_content):
    """计算文件哈希值，现在md5已被破解，生产勿用"""
    return hashlib.md5(file_content).hexdigest()

def get_file_info(uploaded_file):
    """获取文件信息"""
    return{
        "name":uploaded_file.name,
        "size":uploaded_file.size,
        "type":uploaded_file.type,
        "upload_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def format_file_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def is_supported_file(filename):
    """检查文件格式"""
    supported=['.pdf', '.docx', '.txt', '.md', '.csv', '.xlsx', '.xls']
    ext = os.path.splitext(filename)[1].lower()
    return ext in supported
