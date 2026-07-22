//决策树，使用类别特征，不经过one-hot转换,  测试保存模型的路径
val sqlStmt1 = "select * from temp_bi.sumt_anls_eigen_value_1"
val positiveSample = sqlContext.sql(sqlStmt1)

val sqlStmt2 = "select * from temp_bi.sumt_anls_eigen_value_0"
val negativeSample = sqlContext.sql(sqlStmt2)

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import java.math.BigDecimal

def featureTransformV2(data: DataFrame, label: Int) = {
val cataValueMap: Map[String, Int] = Map("1_女装"->1, "2_男装"->2, "3_内衣"->3, "4_鞋类"->4, "5_箱包"->5,
	"6a_彩妆"->6, "6b_护肤"->7, "7_亲子"->8, "7a_奶粉"->9, "8_体用户外"->10,
	"9_家电"->11, "9a_手机通讯"->12, "10_家居家纺"->13, "11_精品"->14, "11a_珠宝饰品"->15,
	"11b_钟表"->16, "12_食品"->17, "13_汽车用品"->18, "14_实体票券"->19, "15_图书"->20,
	"多品类"->21, "16其他"->22, "无"->23)
val vmarkValueMap: Map[String, Int] = Map("皇冠"->1, "钻石"->2, "金牌"->3, "银牌"->4, "铜牌"->5, "铁牌"->6)
val fstSourceValueMap: Map[String, Int] = Map("mobile"->0, "pc"->1, "unknown"->1)  
val rdd = data.map( a=> {
  val dimension = 24
  val featureArray:Array[Double] = new Array[Double](dimension) 
  featureArray(0) = a.getDouble(5)
  featureArray(1) = a.getDouble(6)
  for(i <- 2 to 8){
	featureArray(i) = a.getLong(i + 6).toDouble
  }
  featureArray(9) = a.getDouble(15)
  featureArray(10) = a.getDecimal(16).doubleValue()
  featureArray(11) = a.getLong(17).toDouble
  featureArray(12) = a.getDouble(18)
  featureArray(13) = a.getInt(20).toDouble 
  val goods_cate:String = a.getString(21).trim()
  featureArray(14) = cataValueMap(goods_cate)-1 
  featureArray(15) = a.getLong(22).toDouble
  featureArray(16) = a.getInt(23).toDouble
  featureArray(17) = a.getInt(24).toDouble
  featureArray(18) = a.getInt(25).toDouble
  val vmark_name = a.getString(26).trim()
  featureArray(19) = vmarkValueMap(vmark_name)-1 
  featureArray(20) = a.getLong(27).toDouble
  val max_hist_goods_cate = a.getString(28).trim()
  featureArray(21) = cataValueMap(max_hist_goods_cate)-1 
  featureArray(22) = a.getDouble(29)
  val fst_source = a.getString(30).trim()
  featureArray(23) = fstSourceValueMap(fst_source)
  val vector = new DenseVector(featureArray)
  LabeledPoint(label, vector)
})
rdd
}

val allPositive = featureTransformV2(positiveSample, 1)
val allNegative = featureTransformV2(negativeSample, 0)
val rdd = allPositive.union(allNegative)

val numClasses = 2
val categoricalFeaturesInfo: Map[Int, Int] = Map(14->23, 19->6, 21->23)
val impurity = "gini"
val maxBins = 32

val maxDepth = 3
val Array(trainingData, testData) = rdd.randomSplit(Array(0.7, 0.3))
val dtModel = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
dtModel.save(sc, "hdfs:/user/u_bi_lv1/decisonTreeModel")
