QT += widgets
CONFIG += c++17

SOURCES += \
    src/main.cpp \
    src/xgbooster.cpp \
    src/mainwindow.cpp

HEADERS += \
    include/xgboost/c_api.h \
    src/xgbooster.hpp \
    src/mainwindow.hpp

INCLUDEPATH += include
LIBS += -L$$PWD/lib -lxgboost