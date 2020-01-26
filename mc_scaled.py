import sys, math
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from operator import add
import numpy as np
import argparse
import logging
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S',File='logs/Log_${yyyy-MM-dd_HH_mm_ss}.log')
logging.getLogger().setLevel(logging.INFO)

'''

This is a pyspark implementation of markov clustering. All the functions here are pyspark version of the functions provided in the below link.
https://github.com/GuyAllard/markov_clustering/blob/master/markov_clustering/mcl.py

'''

def matrix_multiply(A, B):

    '''
    
    This function returns the cross product between two matrices represented in Coordinate matrix format
    It is implemented by making simple joins. The code is implemented by refering to the scala implementation in the below link
    https://medium.com/balabit-unsupervised/scalable-sparse-matrix-multiplication-in-apache-spark-c79e9ffc0703

    A: CoordinateMatrix Dataframe
    B: CoordinateMatrix Dataframe
    returns: CoordinateMatrix Dataframe of cross product between A and B

    '''

    A_rdd = A.entries.map(lambda x: (x.j,(x.i,x.value))) # Convert dataframe to rdd of (column,(row, value))
    B_rdd = B.entries.map(lambda x: (x.i,(x.j,x.value))) # Convert dataframe to rdd of (row,(column, value))

    interm_rdd = A_rdd.join(B_rdd).map(lambda x: ((x[1][0][0],x[1][1][0]),(x[1][0][1]*x[1][1][1]))) # Join two rdds and convert to ((row,column),(value))
    C_rdd = interm_rdd.reduceByKey(add).map(lambda x: MatrixEntry(x[0][0],x[0][1],x[1])) # Add the product of same (row,column) pair and convert each row into a matrix entry of (row, column, value)
    return CoordinateMatrix(C_rdd)

def matrix_multiply_mod(a, b):

    '''

    This functions returns the cross product between two matrices represented in Coordinate matrix format
    The only difference is, this function is implemented in BlockMatrix style to speed up the process for small matrices
    This function might fail when we scale up the size of the matrix as it might convert the df into a dense matrix format internally

    a: CoordinateMatrix Dataframe
    b: CoordinateMatrix Dataframe
    returns: CoordinateMatrix Dataframe of cross product between a and b

    '''
    bmat_a = a.toBlockMatrix()
    b_tanspose= b.transpose()
    bmat_b_tanspose=b_tanspose.toBlockMatrix()
    bmat_result= bmat_a.multiply(bmat_b_tanspose)
    return bmat_result.toCoordinateMatrix()


def normalize_mat(df):

    '''

    Calculate L1 norm of a matrix represented as a coordinate matrix in dataframe

    df: Dataframe with 3 columns: source, destination, and weight
    returns: Dataframe of L1 norm of the matrix in coordinate matrix style (source, destination, and weight)

    '''
    cols = df.columns
    df = df.withColumnRenamed(cols[0],'src').withColumnRenamed(cols[1],'dest').withColumnRenamed(cols[2],'wt') # Rename columns temporarily

    tdf = df.groupby('dest').agg({'wt':'sum'}).withColumnRenamed('dest','dest_t').withColumnRenamed('sum(wt)','total_t') # Calculate sum of each column in the matrix
    df = df.join(tdf,df.dest==tdf.dest_t) # Join back the sum to the correspondong elements
    df = df.withColumn('new_wts', col('wt').cast('float')/col('total_t')) # Normalize the weight
    df = df.select('src','dest','new_wts') # Select the new weights
    df = df.withColumnRenamed('src',cols[0]).withColumnRenamed('dest',cols[1]).withColumnRenamed('new_wts',cols[2]) # Rename to the original columns
    return df

def expand_mat(df,power,blockstyle=True):

    '''

    Calculate nth power of a matrix A - A^n

    df: Dataframe of the coordinate matrix A
    power: Integer n. Exponent to which the matrix should be raised
    blockstyle: Boolean. Calculate matrix multiplication block style or by simple rdd joins
    returns: Dataframe of A^n matrix with source, destination, and weight columns

    '''

    # Convert into CoordinateMatrix
    cols = df.columns
    cdf =  CoordinateMatrix(df.rdd.map(tuple))
    rdf = cdf

    # Calculate A^n blockstyle or rdd join style
    if blockstyle:
        for i in range(power-1):
            rdf = matrix_multiply_mod(rdf,cdf)
    else:
        for i in range(power-1):
            rdf = matrix_multiply(rdf,cdf)

    # Convert back to dataframe and return
    rdf_rdd = rdf.entries.map(lambda x: (x.i,x.j,x.value))
    result_df = rdf_rdd.toDF()
    result_df = result_df.withColumnRenamed('_1',cols[0]).withColumnRenamed('_2',cols[1]).withColumnRenamed('_3',cols[2])
    return result_df

def inflate_mat(df,inflate_size):

    '''

    Raise each element to the given power

    df: Dataframe of the coordinate matrix
    inflate_size: Integer or Float. power to ehich each element should be raised
    returns: Dataframe of inflated matrix with source, destination, and weight columns

    '''

    cols = df.columns
    df = df.withColumnRenamed(cols[0],'src').withColumnRenamed(cols[1],'dest').withColumnRenamed(cols[2],'wt') # Rename columns temporarily

    df = df.withColumn('new_wts', col('wt')**inflate_size) # Raise each element to the power inflate_size
    df = df.select('src','dest','new_wts') # Select the new weights
    df = df.withColumnRenamed('src',cols[0]).withColumnRenamed('dest',cols[1]).withColumnRenamed('new_wts',cols[2]) # Rename to the original columns
    df = normalize_mat(df) # Normalize the dataframe
    return df

def prune_mat(df,threshold):

    '''

    Prune the matrix if the weights are below a certain threshold

    df: Dataframe of the coordinate matrix
    threshold: Threshold below which weights are ignored
    returns: Pruned Dataframe with source, destination, and weight columns

    '''
    # Filter for the weights above the threshold and return
    cols = df.columns
    df = df.filter(col(cols[2])>threshold)
    return df

def converged(df1,df2):

    '''

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
    #                                                                                    #
    #  THIS FUNCTION IS NOT TESTED YET. THE FUNCTION FAILS TO RECOGNIZE THE CONVERGENCE  #
    #  WHICH CAUSES THE ALGORITHM TO RUN THROUGH ALL THE SPECIFIED NUMBER OF ITERATIONS  #
    #                                                                                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #

    Check for convergence by calculating the difference between the weights and if they are under certain threshold return True else False

    df1: Dataframe of the coordinate matrix 1
    df2: Dataframe of the coordinate matrix 2
    returns: Bool

    '''

    # Rename columns temporarily
    cols1 = df1.columns
    cols2 = df2.columns
    df1 = df1.withColumnRenamed(cols1[0],'src1').withColumnRenamed(cols1[1],'dest1').withColumnRenamed(cols1[2],'wt1').persist()
    df2 = df2.withColumnRenamed(cols2[0],'src2').withColumnRenamed(cols2[1],'dest2').withColumnRenamed(cols2[2],'wt2').persist()
    df1.count()
    df2.count()

    @udf('int')
    def np_allclose(a,b):
        return int(np.allclose(a, b))

    df = df2.join(df1,(df1.src1==df2.src2) & (df1.dest1==df2.dest2), 'left').persist() # Join the tables to compare weights
    df.count()
    df = df.fillna({'wt1':0}) # For the missing edges (since in the pruning we are removing edges below threshold) in the dataframe, fill them with 0
    df = df.withColumn('allclose',np_allclose(col('wt1'),col('wt2'))).persist()

    #df1 = df1.groupby('src').agg(collect_list('wt')).withColumnRenamed('src','src_1').withColumnRenamed('collect_list(wt)','wt_1')
    #df2 = df2.groupby('src').agg(collect_list('wt')).withColumnRenamed('src','src_2').withColumnRenamed('collect_list(wt)','wt_2')

    #df = df1.join(df2,df1.src_1==df2.src_2).persist()
    df.count()
    df1.unpersist()
    df2.unpersist()

    #df = df.withColumn('allclose',np_allclose(col('wt_1'),col('wt_2')))

    # If all the elements are close enough return true else false
    if df.count() == df.filter(df.allclose==1).count():
        df.unpersist()
        return True
    else:
        df.unpersist()
        return False

def get_clusters(df):

    '''

    Once we have the converged matrix dataframe, fetch the clusters from it

    df: Dataframe of the coordinate matrix
    returns: Dataframe of the clusters

    '''

    cols = df.columns
    df = df.withColumnRenamed(cols[0],'src').withColumnRenamed(cols[1],'dest').withColumnRenamed(cols[2],'wt') # Rename columns temporarily

    # For each row of non-zero diagonal element, get the non-zero columns and group them to form a cluster
    diagonals = df.filter((df.src==df.dest)&(df.wt>0)).select('src').distinct().collect() # Get non-zero diagonals
    ids = [r[0] for r in diagonals]
    fdf = df.filter(df.src.isin(ids)).groupby('src').agg(collect_list('dest')).withColumnRenamed('collect_list(dest)','clusters') # Filter the rows and collect non-zero columns
    fdf = fdf.rdd.zipWithIndex().toDF().withColumnRenamed('_1','nodes_in_cluster').withColumnRenamed('_2','cluster_id') # Give an index for each cluster and rename them sccordingly
    fdf = fdf.select('cluster_id','nodes_in_cluster')
    return fdf

def run_scaled_mcl(sqlContext, matrix, expansion=2, inflation=2, loop_value=1,iterations=100, pruning_threshold=0.001, pruning_frequency=1, convergence_check_frequency=1):

	
	
	pass
	
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-inp', '--input', help="HDFS path of input dataframe")
	parser.add_argument('-out', '--output', help="HDFS path for storing the output cluters")
	parser.add_argument('-e', '--expansion', type=int, help="Expansion rate (int)", defualt = 2)
	parser.add_argument('-r', '--inflation', type=int, help="Inflation rate (int)", defualt = 2)
	parser.add_argument('-l', '--loop_value', type=int, help="Value for self loops (int)", defualt = 1)
	parser.add_argument('-i', '--iterations', type=int, help="Number of iterations", defualt = 100)
	parser.add_argument('-p', '--pruning_threshold', type=float, help="Pruning threshold", defualt = 0.001)
	parser.add_argument('-f', '--pruning_frequency', type=int, help="Pruning frequency", defualt = 1)
	parser.add_argument('-c', '--check', type=int, help="Convergence check frequency", defualt = 1)	
	args = parser.parse_args()
	
	sc = SparkContext()
	sqlContext = SQLContext(sc)
	
	df = sqlContext.read.load(args.input)
	result = run_scaled_mcl(sqlContext,df,args.expansion,args.inflation,args.loop_value,args.iterations,args.pruning_threshold,args.pruning_frequency,args.check)
	result.repartition(100).write.format("parquet").save(args.output)
	print('Refer the output path for result)