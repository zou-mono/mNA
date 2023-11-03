pyinstaller -D test.py --hidden-import PyQt5.QtXml --hidden-import PyQt5.QtNetwork --hidden-import PyQt5.QtSql --hidden-import PyQt5.QtPrintSupport --hidden-import PyQt5.QtPositioning

python-qgis-ltr -m PyInstaller .\main.spec

pyinstaller main.spec