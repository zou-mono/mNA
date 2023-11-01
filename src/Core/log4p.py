import datetime
import logging
import inspect
import os
import sys
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
import warnings

import colorlog
# from PyQt5.QtCore import QObject, pyqtSignal
# from PyQt5.QtWidgets import QPlainTextEdit, QMessageBox

# from UICore.Gv import SplitterState, singleton
from colorama import Fore
from tqdm import tqdm

from Core.common import singleton

warnings.filterwarnings("ignore", category=RuntimeWarning)  # 忽略warning
warnings.filterwarnings("ignore", category=FutureWarning)

cur_path, filename = os.path.split(os.path.abspath(sys.argv[0]))
log_path = os.path.join(cur_path, 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)  # 如果不存在这个logs文件夹，就自动创建一个


# 文件的命名
# c_logName = os.path.join(log_path, '%s.log' % (os.path.basename(os.path.realpath(sys.argv[0])).split('.')[0] + '_' + time.strftime('%Y-%m-%d-%H-%M-%S')))
c_logName = os.path.join(log_path, '%s.log' % ("oneTools" + '_' + time.strftime('%Y-%m-%d-%H-%M-%S')))


log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red'
}


@contextmanager
def tqdm_output(tqdm, write=sys.stderr.write):
    def wrapper(message):
        if message != '\n':
            tqdm.clear()
        write(message)
        if '\n' in message:
            tqdm.display()


class mTqdm(tqdm):
    def __init__(self, iteration=None, ncols=100, desc=None, leave=True, position=0, total=None):
        super(mTqdm, self).__init__(
            iterable=iteration,
            ncols=ncols,
            leave=leave,
            position=position,
            desc=desc,
            total=total,
            bar_format='%s{l_bar}{bar}%s{r_bar}' % (Fore.BLUE, Fore.BLUE))


# class Handler(QObject, logging.Handler):
#     new_record = pyqtSignal(object)
#     clear_record = pyqtSignal(object)
#     set_record = pyqtSignal(object)
#
#     def __init__(self, parent):
#         # super().__init__(parent)
#         # super(logging.Handler).__init__()
#         QObject.__init__(self)
#         logging.Handler.__init__(self)
#         self.parent = parent
#         self.stringList = []
#         self.setLevel(logging.INFO)  # logviewer不显示info以下级别的日志
#         # formatter = logging.Formatter(
#         #     '[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s]- %(message)s')
#         # formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]- %(message)s')
#         formatter = logging.Formatter('[%(asctime)s] - %(message)s')
#         self.setFormatter(formatter)
#         if hasattr(parent, "splitter"):
#             parent.splitter.handle(1).handleClicked.connect(self.handleClicked)
#         elif hasattr(parent, "dockLogView"):
#             q = parent.dockLogView.toggleViewAction()
#             if q is not None:
#                 q.triggered.connect(self.handleClicked)
#
#         self.color = "#000000"
#
#     def setColor(self, _color):
#         self.color = _color
#
#     def handleClicked(self):
#         pass
#         # if hasattr(self.parent, "splitter"):
#         #     if self.parent.splitter.splitterState == SplitterState.collapsed:
#         #         #  splitter缩起来时必须要先清空plainText，否则会卡死
#         #         self.clear_record.emit(self.parent)
#         #     else:
#         #         # output = os.linesep.join(self.stringList)
#         #         output = "<br>".join(self.stringList)
#         #         self.set_record.emit(output)
#         # elif hasattr(self.parent, "dockLogView"):
#         #     if not self.parent.dockLogView.isVisible():
#         #         self.clear_record.emit(self.parent)
#         #     else:
#         #         output = "<br>".join(self.stringList)
#         #         self.set_record.emit(output)
#
#     def emit(self, record):
#         msg = self.format(record)
#         if record.levelno == 40:
#             msg = '<font color=\"#FF0000\">{}</font>'.format(msg)  ## 红色
#         elif record.levelno == 30:
#             msg = '<font color=\"#FFA500\">{}</font>'.format(msg)  ## 橙色
#         elif record.levelno == 20:
#             msg = '<font color={}>{}</font>'.format(self.color, msg)  ## 自定义颜色
#         # msg = "<span style=\" font-size:8pt; font-weight:600; color:#ff0000;\" >"
#         # msg += self.format(record)
#         # msg += "</span>"
#
#         if hasattr(self.parent, "splitter"):
#             if self.parent.splitter.splitterState == SplitterState.expanded:
#                 if len(self.stringList) > 500:
#                     self.clear_record.emit(self.parent)
#                 else:
#                     self.new_record.emit(msg)  # <---- emit signal here
#         elif hasattr(self.parent, "dockLogView"):
#             if self.parent.dockLogView.isVisible():
#                 if len(self.stringList) > 500:
#                     self.clear_record.emit(self.parent)
#                 else:
#                     self.new_record.emit(msg)  # <---- emit signal here
#
#         if len(self.stringList) > 500:
#             self.stringList.clear()
#         self.stringList.append(msg)


class up_stacked_logger:
    def __init__(self, logger, n):
        self.logger = logger

        calling_frame = inspect.stack()[n+1].frame
        trace = inspect.getframeinfo(calling_frame)

        class UpStackFilter(logging.Filter):
            def filter(self, record):
                record.filename = Path(trace.filename).name
                record.lineno = trace.lineno
                record.pathname = trace.filename
                record.module = trace.filename
                record.funcName = trace.function
                return True

        self.f = UpStackFilter()

    def __enter__(self):
        self.logger.addFilter(self.f)
        return self.logger

    def __exit__(self, *args, **kwds):
        self.logger.removeFilter(self.f)

@singleton
class Log:
    def __init__(self, fileName=None):
        # logging.setLoggerClass(WrappedLogger)
        # self.logName = os.path.join(log_path, '%s.log' % (os.path.basename(fileName).split('.')[0] + '_' + time.strftime('%Y-%m-%d-%H-%M-%S')))
        self.logName = c_logName
        self.fileName = fileName
        self.handle_logs()
        self.handler = None

    # def setLogViewer(self, parent, logViewer: QPlainTextEdit):
    #     self.handler = Handler(parent)
    #     self.logViewer = logViewer
    #     self.handler.new_record.connect(self.logViewer.appendHtml)
    #     self.handler.clear_record.connect(self.logViewer.clear)
    #     self.handler.set_record.connect(self.logViewer.appendHtml)

    def get_file_sorted(self, file_path):
        """最后修改时间顺序升序排列 os.path.getmtime()->获取文件最后修改时间"""
        dir_list = os.listdir(file_path)
        if not dir_list:
            return
        else:
            dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
            return dir_list

    def TimeStampToTime(self, timestamp):
        """格式化时间"""
        timeStruct = time.localtime(timestamp)
        return str(time.strftime('%Y-%m-%d', timeStruct))

    def handle_logs(self):
        """处理日志过期天数和文件数量"""
        dir_list = ['logs']  # 要删除文件的目录名
        for dir in dir_list:
            dirPath = os.path.dirname(self.logName)   # 拼接删除目录完整路径
            file_list = self.get_file_sorted(dirPath)  # 返回按修改时间排序的文件list
            if file_list:  # 目录下没有日志文件
                for i in file_list:
                    file_path = os.path.join(dirPath, i)  # 拼接文件的完整路径
                    t_list = self.TimeStampToTime(os.path.getctime(file_path)).split('-')
                    now_list = self.TimeStampToTime(time.time()).split('-')
                    t = datetime.datetime(int(t_list[0]), int(t_list[1]),
                                          int(t_list[2]))  # 将时间转换成datetime.datetime 类型
                    now = datetime.datetime(int(now_list[0]), int(now_list[1]), int(now_list[2]))
                    if (now - t).days > 10:  # 创建时间大于10天的文件删除
                        self.delete_logs(file_path)
                # if len(file_list) > 50:  # 限制目录下记录文件数量
                #     file_list = file_list[0:-50]
                #     for i in file_list:
                #         file_path = os.path.join(dirPath, i)
                #         self.delete_logs(file_path)

    def delete_logs(self, file_path):
        try:
            os.remove(file_path)
        except PermissionError as e:
            Log().warning('删除日志文件失败：{}'.format(e))

    def __console(self, level, message, color):
        self.logger = logging.getLogger(self.fileName)
        self.logger.setLevel(logging.DEBUG)

        if self.handler is not None:
            self.handler.setColor(color)
            self.logger.addHandler(self.handler)

        formatter = logging.Formatter(
            '[%(asctime)s] [%(filename)s:%(funcName)s:%(lineno)d] [%(levelname)s]- %(message)s')  # 日志输出格式
        # 创建一个FileHandler，用于写到本地
        fh = RotatingFileHandler(filename=self.logName, mode='a', maxBytes=1024 * 1024 * 100, backupCount=5,
                                 encoding='utf-8')  # 使用RotatingFileHandler类，滚动备份日志
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # 创建一个StreamHandler,用于输出到控制台
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)s] [%(filename)s:%(funcName)s:%(lineno)d] [%(levelname)s]- %(message)s',
            log_colors=log_colors_config)  # 日志输出格式
        ch = colorlog.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if level == 'info':
            with up_stacked_logger(self.logger, n=2) as logger:
                logger.info(message)
        elif level == 'debug':
            with up_stacked_logger(self.logger, n=2) as logger:
                logger.debug(message)
        elif level == 'warning':
            with up_stacked_logger(self.logger, n=2) as logger:
                logger.warning(message)
        elif level == 'error':
            with up_stacked_logger(self.logger, n=2) as logger:
                logger.error(message)

        # 这两行代码是为了避免日志输出重复问题
        self.logger.removeHandler(ch)
        self.logger.removeHandler(fh)
        if self.handler is not None:
            self.logger.removeHandler(self.handler)
        fh.close()  # 关闭打开的文件

    def debug(self, message, color="#000000"):
        self.__console('debug', message, color)

    def info(self, message, parent=None, color="#000000", dialog=False):
        self.__console('info', message, color)
        # if dialog:
        #     QMessageBox.information(parent, "提示", message, QMessageBox.Close)

    def warning(self, message, parent=None, color="#000000", dialog=False):
        self.__console('warning', message, color)
        # if dialog:
        #     QMessageBox.warning(parent, "警告", message, QMessageBox.Close)

    def error(self, message, parent=None, color="#000000", dialog=False):
        self.__console('error', message, color)
        # if dialog:
        #     QMessageBox.critical(parent, "错误", message, QMessageBox.Close)
