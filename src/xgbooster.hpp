#pragma once

#include <xgboost/c_api.h>
#include <QObject>
#include <QVector>
#include <QString>
#include <QMap>
#include <QHash>
#include <stdexcept>

class XGBModel : public QObject {
    Q_OBJECT
public:
    XGBModel(const QMap<QString, QString>& params, QObject* parent = nullptr);
    virtual ~XGBModel();

    virtual void Fit(const QVector<QVector<double>>& X,
                 const QVector<double>& y,                
                 float startProgressValue = 0.0f,
                 float endProgressValue = 1.0f) = 0;


    virtual QVector<double> Predict(const QVector<QVector<double>>& X) = 0;

    void SaveModel(const QString& filename);
    void LoadModel(const QString& filename);

    void setTerminated(bool flag) { terminated_ = flag; }
    bool isTerminated() const { return terminated_; }

signals:
    void progress(float value);

protected:
    BoosterHandle booster_ = nullptr;
    DMatrixHandle dtrain_ = nullptr;
    QMap<QString, QString> params_;
    int n_features_ = 0;
    bool terminated_ = false;

    void CreateDMatrix(const QVector<QVector<double>>& X, DMatrixHandle& dmat);
    void SetBoosterParams();
};

class XGBRegressor : public XGBModel {
public:
    XGBRegressor(const QMap<QString, QString>& params, QObject* parent = nullptr);
    void Fit(const QVector<QVector<double>>& X,
         const QVector<double>& y,       
         float startProgressValue = 0.0f,
         float endProgressValue = 1.0f) override;
    QVector<double> Predict(const QVector<QVector<double>>& X) override;
};

class XGBClassifier : public XGBModel {
public:
    XGBClassifier(const QMap<QString, QString>& params, QObject* parent = nullptr);
      // Переопределяем виртуальную функцию базового класса
    void Fit(const QVector<QVector<double>>& X,
             const QVector<double>& y,
             float startProgressValue = 0.0f,
             float endProgressValue = 1.0f) override;

    // Добавляем новую версию с stabilizer как отдельную функцию (не override)
    void Fit(const QVector<QVector<double>>& X,
             const QVector<double>& y,
             const QVector<float>& stabilizer,
             float startProgressValue = 0.0f,
             float endProgressValue = 1.0f);
    QVector<double> Predict(const QVector<QVector<double>>& X) override;

private:
    QHash<double, int> label_to_index_;
    QVector<double> index_to_label_;
    QVector<float> EncodeLabels(const QVector<double>& y);
    QVector<double> DecodeLabels(const QVector<float>& pred);
};
