// mainwindow.cpp
#include "mainwindow.hpp"

#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>
#include <QHeaderView>
#include <random>
#include <algorithm>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent) {

    setWindowTitle("XGBoost Qt Trainer");

    // --- UI Setup ---
    auto central = new QWidget(this);
    setCentralWidget(central);
    auto layout = new QVBoxLayout(central);

    // Load CSV button
    loadButton_ = new QPushButton("Load CSV", this);
    layout->addWidget(loadButton_);
    connect(loadButton_, &QPushButton::clicked, this, &MainWindow::loadCSV);

    // Feature selector table
    featureTable_ = new QTableWidget(this);
    featureTable_->setColumnCount(1);
    featureTable_->setHorizontalHeaderLabels({"Select Features (check)"});
    featureTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    layout->addWidget(featureTable_);

    // Target & Stabilizer selectors
    QHBoxLayout *rowSelectors = new QHBoxLayout;
    targetBox_ = new QComboBox(this);
    stabilizerBox_ = new QComboBox(this);
    taskBox_ = new QComboBox(this);
    taskBox_->addItems({"Regression", "Classification"});

    rowSelectors->addWidget(new QLabel("Target:"));
    rowSelectors->addWidget(targetBox_);
    rowSelectors->addWidget(new QLabel("Stabilizer:"));
    rowSelectors->addWidget(stabilizerBox_);
    rowSelectors->addWidget(new QLabel("Task:"));
    rowSelectors->addWidget(taskBox_);
    layout->addLayout(rowSelectors);

    // Params inputs
    QHBoxLayout *paramsLayout = new QHBoxLayout;
    iterEdit_ = new QLineEdit("10", this);
    depthEdit_ = new QLineEdit("3", this);
    etaEdit_ = new QLineEdit("0.1", this);
    lambdaEdit_ = new QLineEdit("1", this);

    paramsLayout->addWidget(new QLabel("n_iter:"));
    paramsLayout->addWidget(iterEdit_);
    paramsLayout->addWidget(new QLabel("max_depth:"));
    paramsLayout->addWidget(depthEdit_);
    paramsLayout->addWidget(new QLabel("eta:"));
    paramsLayout->addWidget(etaEdit_);
    paramsLayout->addWidget(new QLabel("lambda:"));
    paramsLayout->addWidget(lambdaEdit_);
    layout->addLayout(paramsLayout);

    // Train button & progress bar
    trainButton_ = new QPushButton("Train Model", this);
    layout->addWidget(trainButton_);
    progressBar_ = new QProgressBar(this);
    progressBar_->setRange(0, 100);
    layout->addWidget(progressBar_);

    // Save / Load model buttons
    QHBoxLayout *modelButtons = new QHBoxLayout;
    saveButton_ = new QPushButton("Save Model", this);
    loadModelButton_ = new QPushButton("Load Model", this);
    modelButtons->addWidget(saveButton_);
    modelButtons->addWidget(loadModelButton_);
    layout->addLayout(modelButtons);

    // Predict button
    predictButton_ = new QPushButton("Predict & Save Results", this);
    layout->addWidget(predictButton_);

    // Connections
    connect(trainButton_, &QPushButton::clicked, this, &MainWindow::startTraining);
    connect(saveButton_, &QPushButton::clicked, this, &MainWindow::saveModel);
    connect(loadModelButton_, &QPushButton::clicked, this, &MainWindow::loadModel);
    connect(predictButton_, &QPushButton::clicked, this, &MainWindow::predict);

    // Initially disable buttons except load CSV
    trainButton_->setEnabled(false);
    saveButton_->setEnabled(false);
    loadModelButton_->setEnabled(false);
    predictButton_->setEnabled(false);
}

// --- Slots Implementation ---

void MainWindow::loadCSV() {
    QString filename = QFileDialog::getOpenFileName(this, "Open CSV File", "", "CSV files (*.csv);;All files (*)");
    if (filename.isEmpty())
        return;

    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, "Error", "Cannot open file");
        return;
    }

    QTextStream in(&file);

    // Read header
    if (in.atEnd()) {
        QMessageBox::warning(this, "Error", "Empty file");
        return;
    }
    QString headerLine = in.readLine();
    columnNames_ = headerLine.split(',');

    // Setup UI for column selectors
    targetBox_->clear();
    stabilizerBox_->clear();
    targetBox_->addItems(columnNames_);
    stabilizerBox_->addItem("None");
    stabilizerBox_->addItems(columnNames_);

    featureTable_->setRowCount(columnNames_.size());
    for (int i = 0; i < columnNames_.size(); ++i) {
        auto item = new QTableWidgetItem(columnNames_[i]);
        item->setCheckState(Qt::Unchecked);
        featureTable_->setItem(i, 0, item);
    }

    // Read all data rows
    QVector<QVector<double>> dataRows;
    while (!in.atEnd()) {
        QString line = in.readLine();
        if (line.trimmed().isEmpty()) continue;
        auto parts = line.split(',');
        if (parts.size() != columnNames_.size()) {
            QMessageBox::warning(this, "Error", "Inconsistent number of columns");
            return;
        }
        QVector<double> row;
        bool ok;
        for (auto& p : parts) {
            double val = p.toDouble(&ok);
            if (!ok) val = 0.0;
            row.append(val);
        }
        dataRows.append(row);
    }
    file.close();

    // Save full data in targets/features later
    // Extract features and targets according to user selection on training
    // Enable train button if columns available
    features_.clear();
    targets_.clear();
    stabilizer_.clear();
    features_test_.clear();
    targets_test_.clear();
    stabilizer_test_.clear();

    trainButton_->setEnabled(true);

    QMessageBox::information(this, "CSV Loaded", QString("Loaded %1 rows, %2 columns").arg(dataRows.size()).arg(columnNames_.size()));

    // Save all dataRows in a member variable for splitting later
    // For simplicity, store dataRows in a private member temporarily
    // (We add it as a member QVector<QVector<double>> dataRows_)

    dataRows_ = dataRows;
}

// -------- Training --------

void MainWindow::startTraining() {
    // Validate selection
    QVector<int> featureIndices;
    for (int i = 0; i < featureTable_->rowCount(); ++i) {
        auto item = featureTable_->item(i, 0);
        if (item->checkState() == Qt::Checked)
            featureIndices.append(i);
    }
    if (featureIndices.isEmpty()) {
        QMessageBox::warning(this, "Error", "Select at least one feature");
        return;
    }

    int targetIdx = targetBox_->currentIndex();
    if (targetIdx < 0) {
        QMessageBox::warning(this, "Error", "Select target column");
        return;
    }

    int stabilizerIdx = stabilizerBox_->currentIndex() - 1; // -1 means None selected

    // Extract data and split 66% train, 34% test
    int totalRows = dataRows_.size();
    if (totalRows < 2) {
        QMessageBox::warning(this, "Error", "Not enough data");
        return;
    }

    // Shuffle indices
    QVector<int> indices(totalRows);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    int trainCount = totalRows * 0.66;

    features_.clear();
    targets_.clear();
    stabilizer_.clear();
    features_test_.clear();
    targets_test_.clear();
    stabilizer_test_.clear();

    for (int i = 0; i < totalRows; ++i) {
        int row = indices[i];
        QVector<double> featRow;
        for (int fidx : featureIndices)
            featRow.append(dataRows_[row][fidx]);
        if (i < trainCount) {
            features_.append(featRow);
            targets_.append(dataRows_[row][targetIdx]);
            if (stabilizerIdx >= 0)
                stabilizer_.append(static_cast<float>(dataRows_[row][stabilizerIdx]));
        } else {
            features_test_.append(featRow);
            targets_test_.append(dataRows_[row][targetIdx]);
            if (stabilizerIdx >= 0)
                stabilizer_test_.append(static_cast<float>(dataRows_[row][stabilizerIdx]));
        }
    }

    // Build params map
    QMap<QString, QString> params;
    params["num_boost_round"] = iterEdit_->text();
    params["max_depth"] = depthEdit_->text();
    params["eta"] = etaEdit_->text();
    params["lambda"] = lambdaEdit_->text();

    // Delete old model
    if (model_) {
        model_->deleteLater();
        model_ = nullptr;
    }

    bool isRegression = (taskBox_->currentText() == "Regression");
    if (isRegression) {
        model_ = new XGBRegressor(params, this);
    } else {
        model_ = new XGBClassifier(params, this);
    }

    // Connect progress signal
    connect(model_, &XGBModel::progress, this, &MainWindow::updateProgress);

    // Train with stabilizer if classification
    if (!isRegression && !stabilizer_.isEmpty()) {
        auto *cls = dynamic_cast<XGBClassifier*>(model_);
        cls->Fit(features_, targets_, stabilizer_, 0.0f, 1.0f);
    } else {
        model_->Fit(features_, targets_, 0.0f, 1.0f);
    }

    saveButton_->setEnabled(true);
    loadModelButton_->setEnabled(true);
    predictButton_->setEnabled(true);

    QMessageBox::information(this, "Training", "Training finished.");
}

void MainWindow::updateProgress(float value) {
    int ivalue = static_cast<int>(value * 100);
    progressBar_->setValue(ivalue);
}

// -------- Save/Load Model --------

void MainWindow::saveModel() {
    if (!model_) {
        QMessageBox::warning(this, "Error", "No model to save");
        return;
    }
    QString filename = QFileDialog::getSaveFileName(this, "Save Model", "", "XGB Model (*.model)");
    if (filename.isEmpty())
        return;
    model_->SaveModel(filename);
}

void MainWindow::loadModel() {
    QString filename = QFileDialog::getOpenFileName(this, "Load Model", "", "XGB Model (*.model)");
    if (filename.isEmpty())
        return;

    QMap<QString, QString> dummyParams; // params won't be used here
    bool isRegression = (taskBox_->currentText() == "Regression");
    if (model_) {
        model_->deleteLater();
        model_ = nullptr;
    }
    if (isRegression)
        model_ = new XGBRegressor(dummyParams, this);
    else
        model_ = new XGBClassifier(dummyParams, this);

    model_->LoadModel(filename);
    saveButton_->setEnabled(true);
    loadModelButton_->setEnabled(true);
    predictButton_->setEnabled(true);

    QMessageBox::information(this, "Load Model", "Model loaded.");
}

// -------- Predict --------

void MainWindow::predict() {
    if (!model_) {
        QMessageBox::warning(this, "Error", "No model loaded");
        return;
    }
    if (features_test_.isEmpty()) {
        QMessageBox::warning(this, "Error", "No test data available");
        return;
    }

    QVector<double> preds = model_->Predict(features_test_);

    if (preds.size() != targets_test_.size()) {
        QMessageBox::warning(this, "Error", "Prediction size mismatch");
        return;
    }

    QString filename = QFileDialog::getSaveFileName(this, "Save Predictions CSV", "", "CSV files (*.csv)");
    if (filename.isEmpty())
        return;

    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "Error", "Cannot open file for writing");
        return;
    }
    QTextStream out(&file);
    out << "y_true,y_pred\n";
    for (int i = 0; i < preds.size(); ++i) {
        out << QString::number(targets_test_[i]) << "," << QString::number(preds[i]) << "\n";
    }
    file.close();

    QMessageBox::information(this, "Predict", "Predictions saved.");
}
