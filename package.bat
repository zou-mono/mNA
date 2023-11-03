pyinstaller -D test.py --hidden-import PyQt5.QtXml --hidden-import PyQt5.QtNetwork --hidden-import PyQt5.QtSql --hidden-import PyQt5.QtPrintSupport --hidden-import PyQt5.QtPositioning

pyinstall-qgis.bat "D:\空间模拟\mNA\main.spec"

python-qgis-ltr -m PyInstaller .\main.spec