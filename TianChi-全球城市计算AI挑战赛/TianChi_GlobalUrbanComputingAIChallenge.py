import time
import xgboost as xgb
import pandas as pd
import linora as la


def read_data(path):
    start = time.time()
    t =  pd.read_csv(path).drop(['deviceID', 'userID'], axis=1)
    hour_dict = dict(map(lambda x:['0'+str(x), str(x//10)+'0'] if x<10 else [str(x), str(x//10)+'0'], range(60)))
    t['hour_min'] = t.time.map(lambda x:x[11:13]+':'+hour_dict[x[14:16]])
    t = t.groupby(['stationID', 'hour_min', 'status']).lineID.count().unstack(2, 0).reset_index()
    predict = pd.read_csv('Metro_testB/testB_submit_2019-01-27.csv', parse_dates=['startTime'])
    predict['hour_min'] = predict.startTime.map(lambda x:str(x)[11:16])
    predict = predict.merge(t, on=['stationID', 'hour_min'], how='left').drop(['inNums', 'outNums', 'hour_min'], axis=1).fillna(0)
    predict.columns = ['stationID', 'startTime', 'endTime', 'outNums', 'inNums']
    predict['out_in'] = 0
    predict = pd.concat([predict[['stationID', 'startTime', 'endTime', 'outNums', 'out_in']].rename(columns={'outNums':'nums'}),
                         predict[['stationID', 'startTime', 'endTime', 'inNums']].rename(columns={'inNums':'nums'})], sort=False)
    predict['out_in'] = predict.out_in.fillna(1)
    predict = predict.sort_values(['stationID', 'out_in', 'startTime']).reset_index(drop=True)
    print("read and deal with data '"+path+"' run time: %d min %.2f s" % divmod((time.time() - start), 60))
    return predict

def feature_eda(feature_path, label_path=None):
    t = read_data(feature_path)
    if label_path is not None:
        t = pd.concat([read_data(label_path).nums.rename('label'), t], axis=1)
    t['hours'] = t.startTime.dt.hour
    t['minutes'] = (t.startTime.dt.minute/10).astype('int')
    t['hours_minutes'] = t.hours+t.minutes/10
    t = t.merge(pd.read_csv('Metro_roadMap.csv', index_col=0).sum().rename('station_concat_nums').reset_index(drop=True).reset_index().rename(columns={'index':'stationID'}), on='stationID', how='left')
    t['is_trans'] = t.station_concat_nums.map(lambda x:0 if x<3 else 1)
    t['nums30_mean'] = t.groupby(['stationID', 'out_in']).nums.rolling(3, center=True).mean().fillna(1).values
    t['nums30'] = t.groupby(['stationID', 'out_in']).nums.rolling(7, center=True).apply(lambda x:x[0], raw=True).fillna(0).values
    t['nums20'] = t.groupby(['stationID', 'out_in']).nums.rolling(7, center=True).apply(lambda x:x[1], raw=True).fillna(0).values
    t['nums10'] = t.groupby(['stationID', 'out_in']).nums.rolling(7, center=True).apply(lambda x:x[2], raw=True).fillna(0).values
    t['nums01'] = t.groupby(['stationID', 'out_in']).nums.rolling(7, center=True).apply(lambda x:x[4], raw=True).fillna(0).values
    t['nums02'] = t.groupby(['stationID', 'out_in']).nums.rolling(7, center=True).apply(lambda x:x[5], raw=True).fillna(0).values
    t['nums03'] = t.groupby(['stationID', 'out_in']).nums.rolling(7, center=True).apply(lambda x:x[6], raw=True).fillna(0).values
    t['nums_sum'] = t[['nums30', 'nums20', 'nums10', 'nums', 'nums01', 'nums02', 'nums03']].sum(axis=1)
    t['nums_mean'] = t[['nums30', 'nums20', 'nums10', 'nums', 'nums01', 'nums02', 'nums03']].mean(axis=1)
    t['nums_std'] = t[['nums30', 'nums20', 'nums10', 'nums', 'nums01', 'nums02', 'nums03']].std(axis=1)
    t['nums_cv'] = t.nums_std/(t.nums_mean+0.001)
    for i in ['nums30', 'nums20', 'nums10', 'nums01', 'nums02', 'nums03']:
        t[i+'_nums_rate'] = t[i]/(t.nums+0.001)
        t[i+'_sum_rate'] = t[i]/(t.nums_sum+0.001)
    t = pd.concat([t, la.feature_column.categorical_onehot_binarizer(t.stationID, prefix='station')[0]], axis=1)
    t = pd.concat([t, la.feature_column.categorical_onehot_binarizer(t.hours.astype(str)+t.minutes.astype(str), prefix='hours_minutes')[0]], axis=1)
    return t

if __name__ == "__main__":
    t = pd.DataFrame()
    for r, day in enumerate([('2019-01-05', '2019-01-06'), ('2019-01-12', '2019-01-13'), ('2019-01-19', '2019-01-20')]):
        t0 = feature_eda('Metro_train/record_'+day[0]+'.csv', label_path='Metro_train/record_'+day[1]+'.csv')
        t0['group'] = r
        t = pd.concat([t, t0]).reset_index(drop=True)
    
    predictors = [i for i in t.columns if i not in ['label', 'stationID', 'group', 'startTime', 'endTime', 'hours', 'minutes', 'hours_minutes', 'split_sample']]
    # param = la.param_search.XGBRegressor.RandomSearch(t[predictors], t.label, 'reg:linear',
    #                                                   la.metrics.mean_absolute_error, scoring=12.7, speedy_param=(45000, 0.7))
    
    model = xgb.XGBRegressor(**{'learning_rate': 0.07, 'n_estimators': 800, 'max_depth': 6, 'min_child_weight': 5, 'reg_alpha': 35.0, 'reg_lambda': 0.59, 'subsample': 0.7, 'colsample_bytree': 0.8, 'colsample_bylevel': 0.8, 'gamma': 0.0, 'max_delta_step': 0, 'scale_pos_weight': 1, 'random_state': 27, 'n_jobs': 7, 'objective': 'reg:linear', 'tree_method': 'auto'})
    model.fit(t[predictors], t.label)
    predict = feature_eda('Metro_testB/testB_record_2019-01-26.csv', label_path=None)
    predict['inNums'] = pd.Series(model.predict(predict[predictors])).clip(0).round()
    predict = predict[['stationID', 'startTime', 'endTime', 'inNums', 'out_in']]
    predict = pd.concat([predict.query('out_in==1').reset_index(drop=True), predict.query('out_in==0').inNums.rename('outNums').reset_index(drop=True)], axis=1)
    predict.drop(['out_in'], axis=1).to_csv('predict.csv', index=False)
