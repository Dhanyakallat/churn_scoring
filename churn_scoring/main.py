import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('prj1').getOrCreate()
from pyspark.sql.functions import *
from pyspark.sql import Window
from pyspark.sql.functions import rank,dense_rank
import datetime
from time import  strftime


def read_source_data():
    customer_df = spark.read.csv('../source_data/customer.csv',inferSchema=True,header=True)
    product_df = spark.read.csv('../source_data/product.csv',inferSchema=True,header=True)
    customer_product_df = spark.read.csv('../source_data/customer_product.csv',inferSchema=True,header=True)
    customer_survey_df = spark.read.csv('../source_data/customer_survey.csv',inferSchema=True,header=True)
    customer_care_df = spark.read.csv('../source_data/customer_care.csv',inferSchema=True,header=True)

    return customer_df, product_df, customer_product_df, customer_survey_df, customer_care_df

def process_customer_survey(customer_survey_df):
    final_customer_survey_df = customer_survey_df.select('customer_id', when(customer_survey_df.rating < 2, 'A')
                                                         .when(customer_survey_df.rating < 3, 'B')
                                                         .when(customer_survey_df.rating < 4, 'C')
                                                         .otherwise('D').alias('churn_grade')).withColumn('source',
                                                                                                          lit("survey"))
    return final_customer_survey_df

def process_customer_care(customer_care_df):
    window_spec = Window.orderBy(desc("dur"))
    selected_customer_care_df = customer_care_df.filter('reason="complaint"').withColumn("rank",
                                                                                         dense_rank().over(window_spec))
    selected_customer_care_df = selected_customer_care_df.withColumn('churn_grade',
                                                                     when(selected_customer_care_df.rank <= 5, 'A')
                                                                     .when(selected_customer_care_df.rank <= 10, 'B')
                                                                     .when(selected_customer_care_df.rank <= 15, 'C')
                                                                     .otherwise('D').alias('churn_grade')).withColumn(
        'source', lit("cusomer_care"))
    final_customer_care_df = selected_customer_care_df.select('customer_id', 'churn_grade', 'source')

    return final_customer_care_df

def process_customer_tenure(customer_product_df):
    customer_tenure_grade_a_df = customer_product_df.filter("(contract_type='Month-to-month') AND (tenure < 3)") \
        .select('customer_id') \
        .withColumn('grade', lit("A"))

    customer_tenure_grade_b_sel_df = customer_product_df.filter("(contract_type ='One year') AND (tenure < 12) ") \
        .withColumn('remaining_months', round(
        months_between(from_unixtime(unix_timestamp('contract_end_date', 'dd/MM/yyyy')), current_date()), 0).cast(
        'integer')) \
        .select('customer_id', 'remaining_months')

    customer_tenure_grade_b_df = customer_tenure_grade_b_sel_df.filter("(remaining_months < 6)") \
        .select('customer_id').withColumn('grade', lit("B"))

    customer_tenure_grade_df = customer_tenure_grade_a_df.union(customer_tenure_grade_b_df).withColumn('source',
                                                                                                       lit('tenure'))
    return customer_tenure_grade_df

def churn_dec(grade, count1):
    churn_temp = when((grade == 'A') & (count1 == 3), 1) \
                     .when((grade == 'A') & (count1 == 2), 2) \
                     .when((grade == 'A') & (count1 == 1), 3) \
                     .when((grade == 'B') & (count1 == 3), 4) \
                     .when((grade == 'B') & (count1 == 2), 5) \
                     .when((grade == 'B') & (count1 == 1), 6) \
                    .otherwise(7)
    return (churn_temp)

def compute_churn_value(final_customer_care_df, final_customer_survey_df, customer_tenure_grade_df):
    final_df = final_customer_care_df.union(final_customer_survey_df).union(customer_tenure_grade_df)

    final_level1_df = final_df.groupBy('customer_id', 'churn_grade').agg(count('*').alias('count_grade'))

    final_output_df = final_level1_df.withColumn('churn_value',
                                                 lit(churn_dec(final_level1_df.churn_grade,
                                                               final_level1_df.count_grade))) \
        .select('customer_id', 'churn_value')

    window_spec = Window.partitionBy("customer_id").orderBy("customer_id", "churn_value")
    output_df = final_output_df.withColumn("rank", dense_rank().over(window_spec)).filter('rank = 1').select(
        'customer_id', 'churn_value')
    return output_df


def write_output(output_df):
    output_df.write.csv('C:\DH\project\churn_output\churn_score.csv')



customer_df, product_df, customer_product_df, customer_survey_df, customer_care_df = read_source_data()
final_customer_survey_df = process_customer_survey(customer_survey_df)
final_customer_care_df = process_customer_care(customer_care_df)
customer_tenure_grade_df = process_customer_tenure(customer_product_df)

output_df = compute_churn_value(final_customer_care_df, final_customer_survey_df, customer_tenure_grade_df)

write_output(output_df)


#if __name__ == '__main__':
    #run_main()


