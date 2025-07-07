// mainwindow.hpp
#pragma once

#include <QMainWindow>
#include <QVector>
#include <QString>
#include <QProgressBar>
#include <QComboBox>
#include <QPushButton>
#include <QLineEdit>
#include <QTableWidget>
#include "xgbooster.hpp"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);

private slots:
    void loadCSV();
    void startTraining();
    void saveModel();
    void loadModel();
    void predict();
    void updateProgress(float value);

private:
    QVector<QVector<double>> features_, features_test_;
    QVector<double> targets_, targets_test_;
    QVector<float> stabilizer_, stabilizer_test_;
    QVector<QVector<double>> dataRows_;
    QStringList columnNames_;

    QComboBox *taskBox_, *targetBox_, *stabilizerBox_;
    QTableWidget *featureTable_;
    QLineEdit *iterEdit_, *depthEdit_, *etaEdit_, *lambdaEdit_;
    QPushButton *loadButton_, *trainButton_, *saveButton_, *loadModelButton_, *predictButton_;
    QProgressBar *progressBar_;

    XGBModel *model_ = nullptr;
};