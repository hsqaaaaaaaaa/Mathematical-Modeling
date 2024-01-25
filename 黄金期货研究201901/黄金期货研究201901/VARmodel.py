# encoding: gbk
import pandas as pd
import numpy as np
import arrow
import re
import matplotlib.pyplot as plt
import time
# 统计包
# from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.diagnostic


'''黄金建模专用脚本'''
# 伦敦金万德获取代码
# w_wsd_data = vba_wsd("SPTAUUSDOZ.IDC","close","2016-01-01","2019-01-11", "TradingCalendar=SHFE", w_wsd_codes, w_wsd_fields, w_wsd_times, w_wsd_errorid)
# 上期黄金获取代码
# w_wsd_data = vba_wsd("AU1606.SHF,AU1612.SHF,AU1706.SHF,AU1712.SHF,AU1806.SHF,AU1812.SHF,AU1906.SHF","close","2016-01-01","2019-01-11", "TradingCalendar=SHFE", w_wsd_codes, w_wsd_fields, w_wsd_times, w_wsd_errorid)
# 上期黄金成交量获取代码
# w_wsd_data = vba_wsd("AU1606.SHF,AU1612.SHF,AU1706.SHF,AU1712.SHF,AU1806.SHF,AU1812.SHF,AU1906.SHF","volume","2016-01-01","2019-01-11", "TradingCalendar=SHFE", w_wsd_codes, w_wsd_fields, w_wsd_times, w_wsd_errorid)
# 常量
# 文件名
shfeDataFileName = 'shfeAu20160101_20190111.csv'
shfeIndexDataFileName = 'SHFEAuIndex.csv'
xauUsdDataFileName = 'xauusd20160101_20190111.csv'
forexCNYUSDFileName = 'CNYUSDforex.csv'
# 综合月
adjMonthDict = {'5月':5, '11月':11}
# 合约顺序
ctrOrderList = ['AU1606.SHF','AU1612.SHF','AU1706.SHF','AU1712.SHF','AU1806.SHF','AU1812.SHF','AU1906.SHF']
volOrderList = ['AU1606.VOL','AU1612.VOL','AU1706.VOL','AU1712.VOL','AU1806.VOL','AU1812.VOL','AU1906.VOL']
# 列名
dateTimeName = 'DateTime'
indexPriceName = 'indexPrice'
xauusdPriceName = 'CLOSE'
# 先处理数据
# 品种指数编制：参照中信建投期货的主力合约换月规则，拼接数据
class shfeFuturesIndexGenerate():
    def __init__(self,dataFileName=str(),ctrOrder=ctrOrderList,volOrder=volOrderList):
        self.data = pd.read_csv(dataFileName,encoding='gbk')
        self.ctrOrderList = ctrOrder
        self.volOrderList = volOrder
    def dataCombine(self):
        # 按合约顺序提取和拼接数据
        catIndexData = {dateTimeName:[],indexPriceName:[]}
        # 根据价格列名进行循环
        j = 0
        for i in range(len(ctrOrderList)):
            # # 按照合约顺序开始抓合约价格数据和日期数据
            # tempdata = self.data.loc[:,[dateTimeName,self.ctrOrderList[i]]]
            # 目前处理的是合约名字
            tempCatName = self.ctrOrderList[i]
            tempVolName = self.volOrderList[i]
            # 获得当前合约的交割月
            settleMonth = int(re.sub("\D","",tempCatName)[-2:])
            # 获得当前合约的年份
            settleYear = float(re.sub("\D","",tempCatName))/100
            settleYear = round(settleYear)
            # 日期循环
            while True:
                # 抓日期和价格
                tempDate = self.data[dateTimeName][j]
                tempPrice = self.data[tempCatName][j]
                # 抓日期的月份
                tempDateMonth = arrow.get(tempDate).month
                # 抓日期的年份
                tempDateYear = int(str(arrow.get(tempDate).year)[-2:])
                # 对月份和年份进行判断，看是否要换月
                if tempDateMonth >= settleMonth:
                    if tempDateYear >= settleYear:
                        break
                # 对月份进行判断，看是否需要加权
                if tempDateMonth == adjMonthDict['5月'] or tempDateMonth == adjMonthDict['11月']:
                    # 目前合约的价格和成交量
                    tempPrice1 = tempPrice
                    tempVol1 = self.data[tempVolName][j]
                    # 抓下一个合约的价格和成交量
                    catName2 = self.ctrOrderList[i + 1]
                    volName2 = self.volOrderList[i + 1]
                    tempPrice2 = self.data[catName2][j]
                    tempVol2 = self.data[volName2][j]
                    # 合成加权平均价格
                    weight1 = tempVol1 / (tempVol1 + tempVol2)
                    weight2 = tempVol2 / (tempVol1 + tempVol2)
                    tempPrice = tempPrice1 * weight1 + tempPrice2 * weight2
                    tempPrice = round(tempPrice,2)
                # 先插入价格
                catIndexData[indexPriceName].append(tempPrice)
                # 最后加入日期
                tempDate = time.strptime(tempDate,'%Y/%m/%d')
                tempDate = time.strftime('%Y/%m/%d',tempDate)
                catIndexData[dateTimeName].append(tempDate)
                # 行数递增
                j += 1
                if j == self.data.shape[0]:
                    break
        catIndexData = pd.DataFrame(catIndexData)
        return catIndexData
    def getNewDateFile(self):
        data = self.dataCombine()
        data.to_csv('SHFEAuIndex.csv',index=False)

# 然后开始建模
class getVarModel():
    # def __init__(self, shfeAuIndexFileName=shfeIndexDataFileName, xauusdFileName=xauUsdDataFileName, inSampleStart='2016-1-4',inSampleEnd='2018-9-28'):

    def __init__(self, shfeAuIndexFileName=shfeIndexDataFileName, xauusdFileName=xauUsdDataFileName,
                 inSampleStart='2018-1-1',inSampleEnd='2018-9-28'):
        self.shfeIndexData = pd.read_csv(shfeAuIndexFileName)
        self.xauusdData = pd.read_csv(xauusdFileName)
        self.forexData = pd.read_csv(forexCNYUSDFileName)
        # 处理数据，用日期做index
        self.shfeIndexData[dateTimeName] = pd.to_datetime(self.shfeIndexData[dateTimeName])
        self.shfeIndexData = self.shfeIndexData.set_index(dateTimeName)
        self.xauusdData[dateTimeName] = pd.to_datetime(self.xauusdData[dateTimeName])
        self.xauusdData = self.xauusdData.set_index(dateTimeName)
        self.forexData[dateTimeName] = pd.to_datetime(self.forexData[dateTimeName])
        self.forexData = self.forexData.set_index(dateTimeName)
        # 样本内数据,构造Series类型数据
        self.shfeIndexInSampleData = self.shfeIndexData[inSampleStart:inSampleEnd]
        self.shfeIndexInSampleData = pd.Series(self.shfeIndexInSampleData[indexPriceName],
                                               index=self.shfeIndexInSampleData.index)
        self.xauusdDataInSampleData = self.xauusdData[inSampleStart:inSampleEnd]
        self.xauusdDataInSampleData = pd.Series(self.xauusdDataInSampleData[xauusdPriceName],
                                                index=self.xauusdDataInSampleData.index)
        self.forexDataInSampleData = self.forexData[inSampleStart:inSampleEnd]
        self.forexDataInSampleData = pd.Series(self.forexDataInSampleData['OPEN'],
                                               index=self.forexDataInSampleData.index)
        # 样本个数
        self.sampleNum = self.shfeIndexInSampleData.shape[0]

    # 价格走势图
    def price_trend_draw(self):
        # 处理数据,化为人民币和1g
        # changeXAUUSD = np.array(self.xauusdDataInSampleData) / 28.3495 * 6.78
        font1 = {'size':23}
        changeXAUUSD = np.multiply(np.array(self.xauusdDataInSampleData) / 28.3495 , np.array(self.forexDataInSampleData))
        changeXAUUSD = np.round(changeXAUUSD,2)
        shfeXAU = np.array(self.shfeIndexInSampleData)
        # 算相关系数，一定要是pd.Series类型数据才可以
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

    # 对数一阶差分处理
    def logdiff(self,data):
        logData = np.log(data)
        logDiffData = np.diff(logData)
        return logDiffData
    # 数据稳定性检验
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
    # 建立var模型,初次先多次使用，查看aic，bic
    def buildModle(self,dataframe,varLagNum):
        orgMod = sm.tsa.VARMAX(dataframe,order=(varLagNum,0),trend='c',exog=None)
        fitMod = orgMod.fit(maxiter=1000,disp=False)
        print(fitMod.summary())
        resid = fitMod.resid

        result = {'fitMod':fitMod,'resid':resid}
        # 最后返回mod用来做后面一系列检验
        return result
    # VAR系统平稳性检验（代替eviews单位根AR root图）
    def olsCusum(self,resid):
        # 原假设：无漂移（平稳），备择假设：有漂移（不平稳）
        result = statsmodels.stats.diagnostic.breaks_cusumolsresid(resid)
        print(result)
        return result
    # 协整
    def cointTest(self,data1,data2):
        result = sm.tsa.stattools.coint(data1,data2)
        print(result)
    # 脉冲响应
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
        # 画价格走势图
        # self.price_trend_draw()
        # 一阶对数差分,把数据的日期索引留出来
        lnSHFEDiff = self.logdiff(self.shfeIndexInSampleData)
        lnSHFEDiffIndex = self.shfeIndexInSampleData.index[1:]
        lnXAUDiff = self.logdiff(self.xauusdDataInSampleData)
        lnXAUDiffIndex = self.xauusdDataInSampleData[1:]
        # print(len(lnSHFEDiffIndex))
        # print(lnSHFEDiff)
        # adf 一次检验
        # shfeAdfResult = self.adftest(self.shfeIndexInSampleData,4)
        # print(shfeAdfResult)
        # xauAdfResult = self.adftest(self.xauusdDataInSampleData,4)
        # print(xauAdfResult)
        # adf 二次检验
        shfeAdfResult = self.adftest(lnSHFEDiff,10)
        # print(shfeAdfResult)
        xauAdfResult = self.adftest(lnXAUDiff,10)
        # print(xauAdfResult)
        # 建模，1-5的之后项
        # 只能用dataFrame多维格式
        lnDataDict = {'lnSHFEDiff':lnSHFEDiff,'lnXAUDiff':lnXAUDiff}
        lnDataDictSeries = pd.DataFrame(lnDataDict,index=lnSHFEDiffIndex)
        data = lnDataDictSeries[['lnSHFEDiff','lnXAUDiff']]
        # 协整检验
        self.cointTest(lnDataDictSeries['lnSHFEDiff'], lnDataDictSeries['lnXAUDiff'])
        # 建模
        shfexau_xauusdMode = self.buildModle(data,2)
        fitMod = shfexau_xauusdMode['fitMod']
        shfeResid = shfexau_xauusdMode['resid']['lnSHFEDiff']
        xauusdResid = shfexau_xauusdMode['resid']['lnXAUDiff']
        # cusum检验
        # self.olsCusum(shfeResid)
        # 脉冲响应(默认使用乔里斯基正交)
        self.impulseResponse(fitMod)
        # ax = fitMod.impulse_responses(10, orthogonalized=True)
        # 方差风险
        self.varianceDue(data)



if __name__ == '__main__':
    # 生成指数文件
    # indexGenerate = shfeFuturesIndexGenerate(shfeDataFileName)
    # indexGenerate.getNewDateFile()
    # 建模
    varmodel = getVarModel()
    varmodel.runModle()

    # a = '2019/1/11'
    # aMonth = arrow.get(a).month
    # print(aMonth,type(aMonth))
