from Bio import SeqIO
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

"""_This software takes aligned ribosomal gene sequences as input, sequencing error are masked by a cutomized
error rate and partially corrected during merging. The pairwise distance matrix between the reading are 
calculated. The indentical sequences are merged followed by merging between highly similar sequences. The outputs
including the pairwise distance after 0 distance merging , the merging outcome(ids and copynumbers) and aligned sequences in fasta file._
Args:
     input_file (filepath): The fasta file
Output_files:
    1. ceratain readings: pandas csv of ceratain readings:with index,seqID and copy number(C_Num)
    2. ceratain+uncertain readings: pandas csv of ceratain+uncertain readings, with index,seqID and copy number(C_Num)
    3. final_merged_certain: fasta: sequences of ceratain readings
    4. final_merged_certananduncertain: fasta: sequences of ceratain and uncertain readings
    5. Counter_Pd: pandas Counter csv: all readings with SeqID and C_Num after 0 distance merging
    6. fasta: sequences after 0 distance merging
    7. Mdis: pairwise distance matrix

"""
def CreateMatrix(input_file):
    """This function creates a matrix from a given fasta file. 

    Args:
        input_file (filepath): The fasta file
    
    return: M(2d array of stringsy): matrix with M_rows of rows and M_cols of columns.
    """
   
    records = SeqIO.to_dict(SeqIO.parse(input_file, "fasta"))
    ids = list(SeqIO.to_dict(SeqIO.parse(input_file, "fasta")).keys())
    array =[
        y
        for i in range(0,len(ids))
            for y in records[ids[i]].seq]
        
    m = np.array(array)
    M = np.reshape(m, (-1, len(records[ids[1]].seq)))
    M_rows, M_cols = M.shape 
    
    return M,M_rows,M_cols
    
def MarkTerminalGaps(M,M_rows, M_cols):
    """_The function Mark the terminal gaps, gaps'-' within the terminal gaps are masked by '+'.
        Defining terminal gaps: 
        1:from the begining or the end of the readings;
        2: at least 3 in a row;
        3: terminated with the first non-gap character;

    Args:
        M (2d array of stringsy):Sequence matrix, product of CreateMatrix(input_file)
        M_rows (int): :rows of array of strings, product of CreateMatrix(input_file)
        M_cols (int):columns of array of strings, product of CreateMatrix(input_file)
    Returns:
        M (array of strings):Sequence matrix
        M_rows (int ): :rows of array of strings
        M_cols (int):columns of array of strings
        MissInfoRowCoords(int array): coordinates of all the readings with terminal gaps
    """
    ##find all gaps
    gapRowCoords,gapColCoords = np.where(M == '-')
    gapCoords=np.vstack((gapRowCoords,gapColCoords)).T


    #fw:the rows with 1st 3 chars are gaps
    MissInfoRowCoordsF=[
        (i)
    for i,j in gapCoords
        if j==0 and M[i,(j+1)]=='-' and M[i,(j+2)]=='-'
]    

    #Bw:the rows with last 3 chars are gaps
    MissInfoRowCoordsB=[
        (i)
        for i,j in gapCoords
            if j==M_cols-1 and M[i,(j-2)]=='-'and M[i,(j-3)]=='-'
] 

    #conbine Fw and Bw
    MissInfoRowCoords=list(set(MissInfoRowCoordsF + MissInfoRowCoordsB))
        # mask rows with terminal gaps 
    for i in MissInfoRowCoordsF:
        for j in range(0,M_cols):
            if M[i,j]== '-':
                M[i,j] = '+'
            if M[i,j+1]!= '-':
                break
    for i in MissInfoRowCoordsB:
        for j in reversed(range(0,M_cols)):
            if M[i,j]== '-':
                M[i,j] = '+'
            if M[i,j-1]!= '-':
                break
   
    M_rows,M_cols = M.shape
    print(len(MissInfoRowCoords)," rows reads with terminal gaps have been Masked.")
    return M,M_rows,M_cols,MissInfoRowCoords

def ErrorMasking(M,M_rows,M_cols,T):
    """This function masks the error reads by analyzing the majority of readings on the same postions, error rate is  
    one of the Input value.

    Args:
        M (2d array of stringsy):Sequence matrix
        M_rows (int ): :rows of array of strings
        M_cols (int):columns of array of strings
        T (Int): customer defined error rate
    return: 
        n/a: the error masking is done onto the matrix of readings.
    """
    for x in range(0,M_cols):
        ele_array = np.unique(M[:,x])
    
        if len(ele_array) == 1:
                pass
        else:
               for i in ele_array:
                ele_per = ((M[:,x]==i).sum(axis=0))/M_rows
           
                if ele_per > T:
                    pass
                elif ele_per <= T and ((M[:,x]=='-').sum(axis=0))>(M_rows*(9/10)):
                    for j in (np.where(M[:,x]==i)):
                        M[j,x]='+'
                else:
                    for j in (np.where(M[:,x]==i)):
                        M[j,x]='n'
            

def StringMatrixToNumber (M1): 
    
    """_Transfer string array in to Int array. Each base and gaps are represented by an int_:
        1: a
        2: c
        3: t
        4: g
        n: 0
        -: -2
        +: -1

    Args:
        M1 (2d array of strings):

    Returns:
        M2(2d array of int):
    """
    
    M2=M1
    M2[M2 == 'a']="1"
    M2[M2 == 'c']="2"
    M2[M2 == 't']="3"
    M2[M2 == 'g']="4"
    M2[M2 == 'n']="0"
    M2[M2 == '-']="5"#-2
    M2[M2 == '+']="6"#-1
    Int = np.vectorize(lambda x: int(x))
    M2 = Int(M1)   
    M2[M2 == 5]=-2
    M2[M2 == 6]=-1
    print("First three position of number Matrix:",M2[0,0],M2[0,1],M2[0,2],"Datatype:",M2.dtype)
    return M2
    
def CalculatePairWiseDistance_sk(u,v):
    """_customizing calculation formula for sklearn pairwise distance matrixfunction_

    Args:
        u (1d array of strings): 1st row of seq reading
        v (1d array of strings): the row of seq reading to pair with 'u' for distance calculation

    Returns:
        a+b(int):the distanct between u and v
    """    
    a = np.bitwise_and(np.bitwise_and(u > 0,v > 0),(u!=v)).sum(axis=0)
    b = np.bitwise_and(u == -2,v>0).sum(axis=0) + np.bitwise_and(v == -2,u>0).sum(axis=0)
    return a+b

def CalculatePairWiseDistance(M2):
    """_This function generate a counter that records the copy number of the number matrix 
        after indentical merging. When two sequences are indentical, the copyNumber 
        of the first template reading increased by 1 and the copynumber of the second seq reading is marked as 0_
        RepeatSystem: In order to avoid repeative calculation onto the indentical sequences. A repeat switch 
        has been implemented, 
        'n' and '+' correction: when indentical merging happens. The 'n's and '+' on the template seq reads are tried to be 
        corrected by the base from the same position of its merging template. If a 'non-n' base appears on the same position
        'n's and '+'s are corrected by non-n bases
    Args:
        M2 (2d array of int): 

    Returns:
        counter(int array):a counter that record the copy number of the number matrix
        M2(int matrix): int matrix, 'n' and '+' corrected
        # Mdis(fullscale): pairwise distance matrix of all readings including 0 distance, can be returned if needed
        # MergeDicts_Pd#Mdis(dictionary of int): a dictionary of the merging path can be calculated and exported in needed
    """    
    
    M1=M2
    M_rows,M_cols = M2.shape
    # distance matrix with M_rows* M_rows shape
    Mdis = np.full((M_rows,M_rows),-1)
    repeat = np.full(M_rows,1,dtype=bool)
    counter = np.full(M_rows,1,dtype=int)
    #MergeDicts_Pd = pd.DataFrame( {'SeqID_i':[],  'SeqID_j':[]})
#    nmcounter =0
    for i in range(0,M_rows-1):
        if repeat[i]==False:
                continue
        else:
            for j in range(i+1,M_rows):
                if repeat[j]==False:
                     continue
                else:
                    a = np.bitwise_and(np.bitwise_and(M1[i,]>0,M1[j,]>0),(M1[i,]!=M1[j,])).sum(axis=0)
                    b = np.bitwise_and(M1[i,]== -2,M1[j,]>0).sum(axis=0) + np.bitwise_and(M1[j,]== -2,M1[i,]>0).sum(axis=0)
                    Mdis[i,j]= a+b  
                    if a+b == 0:
                        counter[i]+=1
                        counter[j]-=1
                        repeat[j]= False

  #                          for x in range(0, len(M1[i,])):
   #                             if M1[i,x] == 0 or M1[i,x] == -1:
    #                                M1[i,x] = M1[j,x]
     #                               nmcounter += 1

    #1.1 If we want to see How Merge happens unhash the code below
    #df2 = pd.DataFrame([[i,j]], columns=['SeqID_i','SeqID_j'])
    # MergeDicts_Pd = pd.concat([df2, MergeDicts_Pd])
                      #  else:
   # print(nmcounter,'of ns and gaps has been corrected')             
    return counter,M1 #MergeDicts_Pd#Mdis


def reduction_Matrix(M2,counter):
    """_This function remove the merged readings fromt he 2d array of the readings in int dtype based on the counter.  
    The readings with postive copynumbers were returned as output_

    Args:
        M2 (_2d array of int_): 
        counter (_1d array of int_): output of the function'CalculatePairWiseDistance(M2)'

    Returns:
        array2(2d array of int): M2 after removing merged reading.
    """    
    M_rows,M_cols = M2.shape
    dict1={}
    for i,j in enumerate(counter):
        dict1[i]=j
    array2 = np.zeros((len(dict1.keys()),M_cols))
    for i in range(0,len(dict1.keys())):
        if dict1[i]==0:
            continue
        else:
            array2[i]=M2[list(dict1.keys())[i]]
    return array2

def Export(Mdis):
    """function that exporting pairwise distance matrix into csv file, more csvs such as merging path can be exported if necssary.

    Args:
        Mdis (2d array of int), 
        
    """    
    from numpy import savetxt
    savetxt('Dist_Matrix_v2.0_reduction_1e2.csv',Mdis,fmt='%i',delimiter=',')
    #savetxt('Counter1e2.csv',counter,fmt='%i',delimiter=',')
    #savetxt('Merge_list.csv',Merge_list,fmt='%i',delimiter=',')#   continue


def NumberMatrixToString (reducedarray):
    """_This function transfer 2d array of int back to 2d array of strings_

    Args:
        reducedarray (2d array of int): 

    Returns:
        array_rev(2d array of srtings): 
    """    
    rows,cols = reducedarray.shape
    array_rev = np.full((rows,cols),'a')
    for i in range(0,rows):
        for j in range(0,cols):
            if reducedarray[i,j]==1.0:
                array_rev[i,j]='a'
            elif reducedarray[i,j]==2.0:
                array_rev[i,j]='c'
            elif reducedarray[i,j]==3.0:
                array_rev[i,j]='t'
            elif reducedarray[i,j]==4.0:
                array_rev[i,j]='g'
            elif reducedarray[i,j]==0.0:
                array_rev[i,j]='n'
            else:
                array_rev[i,j]= '-'
    return array_rev
    
def StringMatrixToSeq(array_rev):
    """_This funciton removes the ',' of 2d array of string, reduce the array dimention to 1d _
        input: ['a','c','t','g'], output:['actg'] 
    Args:
        array_rev (2d array of strings)

    Returns:
        seq_array(1d array of strings): 
    """    
    rows,cols = array_rev.shape
    separator = ''
    seq_array = np.array([
            separator.join(array_rev[i,])
            for i in range(0,rows)
        ])
    return seq_array

def FastaRecords(seq_array):
    """_This function write 1d array of strings into SeqRecords of SeqIO_, and expor a fasta file

    Args:
        seq_array (1d array of strings)
    """    
    rows, = seq_array.shape
    separator = ''
    a = np.array([seq_array[i].item() for i in range(0,rows)])
    b = [Seq(i) for i in a]
    records = [
           SeqRecord(Seq(seq), id = str(index), description = "") 
           for index,seq in enumerate(b) 
    ]
    SeqIO.write(records, 'Merged_readings' ,"fasta")
    return 

     
def main():
       
    import timeit
    input_file = sys.argv[1]  
    start = timeit.default_timer()
    M,M_rows,M_cols = CreateMatrix(input_file)
    M,M_rows,M_cols = RemoveTerminalGaps(M,M_rows, M_cols)
    ErrorMasking(M,M_rows,M_cols,0.01)
    M2 = StringMatrixToNumber (M)
    counter,Merge_list = CalculatePairWiseDistance(M2)
    array2 = reduction_Matrix(M2,counter)
    reducedarray = array2[~np.all(array2 == 0, axis=1)]
    Mdis = pairwise_distances(reducedarray,metric = CalculatePairWiseDistance_sk)
    Export(Mdis,counter,Merge_list)
    array_rev = NumberMatrixToString (reducedarray)
    seq_array = StringMatrixToSeq(array_rev)
    FastaRecords(seq_array)

    stop = timeit.default_timer()
    execution_time = stop - start

    print("Program Executed in "+str(execution_time))

   
if __name__ == "__main__":
    main()
