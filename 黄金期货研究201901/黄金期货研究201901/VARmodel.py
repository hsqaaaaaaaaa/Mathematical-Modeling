# encoding: gbk
import pandas as pd
import numpy as np
import arrow
import re
import matplotlib.pyplot as plt
import time
# ͳ�ư�
# from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.diagnostic


'''�ƽ�ģר�ýű�'''
# �׶ؽ���»�ȡ����
# w_wsd_data = vba_wsd("SPTAUUSDOZ.IDC","close","2016-01-01","2019-01-11", "TradingCalendar=SHFE", w_wsd_codes, w_wsd_fields, w_wsd_times, w_wsd_errorid)
# ���ڻƽ��ȡ����
# w_wsd_data = vba_wsd("AU1606.SHF,AU1612.SHF,AU1706.SHF,AU1712.SHF,AU1806.SHF,AU1812.SHF,AU1906.SHF","close","2016-01-01","2019-01-11", "TradingCalendar=SHFE", w_wsd_codes, w_wsd_fields, w_wsd_times, w_wsd_errorid)
# ���ڻƽ�ɽ�����ȡ����
# w_wsd_data = vba_wsd("AU1606.SHF,AU1612.SHF,AU1706.SHF,AU1712.SHF,AU1806.SHF,AU1812.SHF,AU1906.SHF","volume","2016-01-01","2019-01-11", "TradingCalendar=SHFE", w_wsd_codes, w_wsd_fields, w_wsd_times, w_wsd_errorid)
# ����
# �ļ���
shfeDataFileName = 'shfeAu20160101_20190111.csv'
shfeIndexDataFileName = 'SHFEAuIndex.csv'
xauUsdDataFileName = 'xauusd20160101_20190111.csv'
forexCNYUSDFileName = 'CNYUSDforex.csv'
# �ۺ���
adjMonthDict = {'5��':5, '11��':11}
# ��Լ˳��
ctrOrderList = ['AU1606.SHF','AU1612.SHF','AU1706.SHF','AU1712.SHF','AU1806.SHF','AU1812.SHF','AU1906.SHF']
volOrderList = ['AU1606.VOL','AU1612.VOL','AU1706.VOL','AU1712.VOL','AU1806.VOL','AU1812.VOL','AU1906.VOL']
# ����
dateTimeName = 'DateTime'
indexPriceName = 'indexPrice'
xauusdPriceName = 'CLOSE'
# �ȴ�������
# Ʒ��ָ�����ƣ��������Ž�Ͷ�ڻ���������Լ���¹���ƴ������
class shfeFuturesIndexGenerate():
    def __init__(self,dataFileName=str(),ctrOrder=ctrOrderList,volOrder=volOrderList):
        self.data = pd.read_csv(dataFileName,encoding='gbk')
        self.ctrOrderList = ctrOrder
        self.volOrderList = volOrder
    def dataCombine(self):
        # ����Լ˳����ȡ��ƴ������
        catIndexData = {dateTimeName:[],indexPriceName:[]}
        # ���ݼ۸���������ѭ��
        j = 0
        for i in range(len(ctrOrderList)):
            # # ���պ�Լ˳��ʼץ��Լ�۸����ݺ���������
            # tempdata = self.data.loc[:,[dateTimeName,self.ctrOrderList[i]]]
            # Ŀǰ������Ǻ�Լ����
            tempCatName = self.ctrOrderList[i]
            tempVolName = self.volOrderList[i]
            # ��õ�ǰ��Լ�Ľ�����
            settleMonth = int(re.sub("\D","",tempCatName)[-2:])
            # ��õ�ǰ��Լ�����
            settleYear = float(re.sub("\D","",tempCatName))/100
            settleYear = round(settleYear)
            # ����ѭ��
            while True:
                # ץ���ںͼ۸�
                tempDate = self.data[dateTimeName][j]
                tempPrice = self.data[tempCatName][j]
                # ץ���ڵ��·�
                tempDateMonth = arrow.get(tempDate).month
                # ץ���ڵ����
                tempDateYear = int(str(arrow.get(tempDate).year)[-2:])
                # ���·ݺ���ݽ����жϣ����Ƿ�Ҫ����
                if tempDateMonth >= settleMonth:
                    if tempDateYear >= settleYear:
                        break
                # ���·ݽ����жϣ����Ƿ���Ҫ��Ȩ
                if tempDateMonth == adjMonthDict['5��'] or tempDateMonth == adjMonthDict['11��']:
                    # Ŀǰ��Լ�ļ۸�ͳɽ���
                    tempPrice1 = tempPrice
                    tempVol1 = self.data[tempVolName][j]
                    # ץ��һ����Լ�ļ۸�ͳɽ���
                    catName2 = self.ctrOrderList[i + 1]
                    volName2 = self.volOrderList[i + 1]
                    tempPrice2 = self.data[catName2][j]
                    tempVol2 = self.data[volName2][j]
                    # �ϳɼ�Ȩƽ���۸�
                    weight1 = tempVol1 / (tempVol1 + tempVol2)
                    weight2 = tempVol2 / (tempVol1 + tempVol2)
                    tempPrice = tempPrice1 * weight1 + tempPrice2 * weight2
                    tempPrice = round(tempPrice,2)
                # �Ȳ���۸�
                catIndexData[indexPriceName].append(tempPrice)
                # ����������
                tempDate = time.strptime(tempDate,'%Y/%m/%d')
                tempDate = time.strftime('%Y/%m/%d',tempDate)
                catIndexData[dateTimeName].append(tempDate)
                # ��������
                j += 1
                if j == self.data.shape[0]:
                    break
        catIndexData = pd.DataFrame(catIndexData)
        return catIndexData
    def getNewDateFile(self):
        data = self.dataCombine()
        data.to_csv('SHFEAuIndex.csv',index=False)

# Ȼ��ʼ��ģ
class getVarModel():
    # def __init__(self, shfeAuIndexFileName=shfeIndexDataFileName, xauusdFileName=xauUsdDataFileName, inSampleStart='2016-1-4',inSampleEnd='2018-9-28'):

    def __init__(self, shfeAuIndexFileName=shfeIndexDataFileName, xauusdFileName=xauUsdDataFileName,
                 inSampleStart='2018-1-1',inSampleEnd='2018-9-28'):
        self.shfeIndexData = pd.read_csv(shfeAuIndexFileName)
        self.xauusdData = pd.read_csv(xauusdFileName)
        self.forexData = pd.read_csv(forexCNYUSDFileName)
        # �������ݣ���������index
        self.shfeIndexData[dateTimeName] = pd.to_datetime(self.shfeIndexData[dateTimeName])
        self.shfeIndexData = self.shfeIndexData.set_index(dateTimeName)
        self.xauusdData[dateTimeName] = pd.to_datetime(self.xauusdData[dateTimeName])
        self.xauusdData = self.xauusdData.set_index(dateTimeName)
        self.forexData[dateTimeName] = pd.to_datetime(self.forexData[dateTimeName])
        self.forexData = self.forexData.set_index(dateTimeName)
        # ����������,����Series��������
        self.shfeIndexInSampleData = self.shfeIndexData[inSampleStart:inSampleEnd]
        self.shfeIndexInSampleData = pd.Series(self.shfeIndexInSampleData[indexPriceName],
                                               index=self.shfeIndexInSampleData.index)
        self.xauusdDataInSampleData = self.xauusdData[inSampleStart:inSampleEnd]
        self.xauusdDataInSampleData = pd.Series(self.xauusdDataInSampleData[xauusdPriceName],
                                                index=self.xauusdDataInSampleData.index)
        self.forexDataInSampleData = self.forexData[inSampleStart:inSampleEnd]
        self.forexDataInSampleData = pd.Series(self.forexDataInSampleData['OPEN'],
                                               index=self.forexDataInSampleData.index)
        # ��������
        self.sampleNum = self.shfeIndexInSampleData.shape[0]

    # �۸�����ͼ
    def price_trend_draw(self):
        # ��������,��Ϊ����Һ�1g
        # changeXAUUSD = np.array(self.xauusdDataInSampleData) / 28.3495 * 6.78
        font1 = {'size':23}
        changeXAUUSD = np.multiply(np.array(self.xauusdDataInSampleData) / 28.3495 , np.array(self.forexDataInSampleData))
        changeXAUUSD = np.round(changeXAUUSD,2)
        shfeXAU = np.array(self.shfeIndexInSampleData)
        # �����ϵ����һ��Ҫ��pd.Series�������ݲſ���
        correlation = round(pd.Series(shfeXAU).corr(pd.Series(changeXAUUSD)),6)
        # print(changeXAUUSD)
        fig = plt.figure(figsize=(12,8))
        plt.plot(changeXAUUSD,'r',label='XAU USD')
        plt.plot(shfeXAU,'g',label='SHFE XAU')
        plt.title('Correlation: ' + str(correlation),font1)
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc=0,prop=font1)
        plt.ylabel('Price',font1)
        plt.show()
        # plt.plot()

    # ����һ�ײ�ִ���
    def logdiff(self,data):
        logData = np.log(data)
        logDiffData = np.diff(logData)
        return logDiffData
    # �����ȶ��Լ���
    def adftest(self,data,maxlags):
        adfResult = sm.tsa.stattools.adfuller(data,maxlags)
        output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used",
                                         "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"],
                                  columns=['value'])
        output['value']['Test Statistic Value'] = adfResult[0]
        output['value']['p-value'] = adfResult[1]
        output['value']['Lags Used'] = adfResult[2]
        output['value']['Number of Observations Used'] = adfResult[3]
        output['value']['Critical Value(1%)'] = adfResult[4]['1%']
        output['value']['Critical Value(5%)'] = adfResult[4]['5%']
        output['value']['Critical Value(10%)'] = adfResult[4]['10%']
        return output
    # ����varģ��,�����ȶ��ʹ�ã��鿴aic��bic
    def buildModle(self,dataframe,varLagNum):
        orgMod = sm.tsa.VARMAX(dataframe,order=(varLagNum,0),trend='c',exog=None)
        fitMod = orgMod.fit(maxiter=1000,disp=False)
        print(fitMod.summary())
        resid = fitMod.resid

        result = {'fitMod':fitMod,'resid':resid}
        # ��󷵻�mod����������һϵ�м���
        return result
    # VARϵͳƽ���Լ��飨����eviews��λ��AR rootͼ��
    def olsCusum(self,resid):
        # ԭ���裺��Ư�ƣ�ƽ�ȣ���������裺��Ư�ƣ���ƽ�ȣ�
        result = statsmodels.stats.diagnostic.breaks_cusumolsresid(resid)
        print(result)
        return result
    # Э��
    def cointTest(self,data1,data2):
        result = sm.tsa.stattools.coint(data1,data2)
        print(result)
    # ������Ӧ
    def impulseResponse(self,fitMod,terms=20):
        font1 = {'size':23}
        ax = fitMod.impulse_responses(terms, orthogonalized=True).plot(figsize=(12, 8))
        plt.legend(prop=font1)
        plt.show()
    def varianceDue(self,dataFrame):
        font1 = {'size': 23}
        md = sm.tsa.VAR(dataFrame)
        re = md.fit(2)
        fevd = re.fevd(10)
        print(fevd.summary())
        fevd.plot()
        plt.legend(prop=font1)
        plt.show()

    def runModle(self):
        # ���۸�����ͼ
        # self.price_trend_draw()
        # һ�׶������,�����ݵ���������������
        lnSHFEDiff = self.logdiff(self.shfeIndexInSampleData)
        lnSHFEDiffIndex = self.shfeIndexInSampleData.index[1:]
        lnXAUDiff = self.logdiff(self.xauusdDataInSampleData)
        lnXAUDiffIndex = self.xauusdDataInSampleData[1:]
        # print(len(lnSHFEDiffIndex))
        # print(lnSHFEDiff)
        # adf һ�μ���
        # shfeAdfResult = self.adftest(self.shfeIndexInSampleData,4)
        # print(shfeAdfResult)
        # xauAdfResult = self.adftest(self.xauusdDataInSampleData,4)
        # print(xauAdfResult)
        # adf ���μ���
        shfeAdfResult = self.adftest(lnSHFEDiff,10)
        # print(shfeAdfResult)
        xauAdfResult = self.adftest(lnXAUDiff,10)
        # print(xauAdfResult)
        # ��ģ��1-5��֮����
        # ֻ����dataFrame��ά��ʽ
        lnDataDict = {'lnSHFEDiff':lnSHFEDiff,'lnXAUDiff':lnXAUDiff}
        lnDataDictSeries = pd.DataFrame(lnDataDict,index=lnSHFEDiffIndex)
        data = lnDataDictSeries[['lnSHFEDiff','lnXAUDiff']]
        # Э������
        self.cointTest(lnDataDictSeries['lnSHFEDiff'], lnDataDictSeries['lnXAUDiff'])
        # ��ģ
        shfexau_xauusdMode = self.buildModle(data,2)
        fitMod = shfexau_xauusdMode['fitMod']
        shfeResid = shfexau_xauusdMode['resid']['lnSHFEDiff']
        xauusdResid = shfexau_xauusdMode['resid']['lnXAUDiff']
        # cusum����
        # self.olsCusum(shfeResid)
        # ������Ӧ(Ĭ��ʹ������˹������)
        self.impulseResponse(fitMod)
        # ax = fitMod.impulse_responses(10, orthogonalized=True)
        # �������
        self.varianceDue(data)



if __name__ == '__main__':
    # ����ָ���ļ�
    # indexGenerate = shfeFuturesIndexGenerate(shfeDataFileName)
    # indexGenerate.getNewDateFile()
    # ��ģ
    varmodel = getVarModel()
    varmodel.runModle()

    # a = '2019/1/11'
    # aMonth = arrow.get(a).month
    # print(aMonth,type(aMonth))
