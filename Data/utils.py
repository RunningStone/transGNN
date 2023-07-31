from transGNN import logger
import pandas as pd
import random
from pathlib import Path

########################################################################
# .    from file to data source
########################################################################

def get_GeneProfileMatrix(      
                            data_file_loc,
                            key='brca.rnaseq',
                            label_file_loc=None,p_id=None,
                            thresh=1.0,
                            with_transpose:bool=False,
                            ):
    tw = tableWorker(data_file_loc)
    tw.read_table(key=key)
    # get patient id and drop score lower than thrh = 1.0
    df2 = tw.df
    # DEBUG:
    df2 = df2.set_index("Unnamed: 0")
    logger.debug(f"{df2[:5]}")
    logger.debug(f"item example: {df2[df2.columns.tolist()[0]]},thresh:{thresh}")
    for name in df2.columns.tolist():
        df2 = df2.drop(df2[df2[name]<thresh].index)
    if with_transpose:
        #  transpose to make  [patient x gene]
        df3 = df2.transpose()
    else:
        df3 = df2
    df3.insert(0,"sample_id",df3.index.to_list())
    df3=df3.reset_index(drop=True)

    if label_file_loc is not None:
        # add patient id for data
        df3["PatientID"] = [name[:12] for name in df3["sample_id"].to_list()]
        # read label file
        tw_dnad = tableWorker(label_file_loc)
        tw_dnad.read_table()
        # add patient id for label
        df_dnad = tw_dnad.df
        df_dnad["PatientID"] = [name[:12] for name in df_dnad[p_id].to_list()]
        # merge to confirm all have label
        df_all = pd.merge(df3,df_dnad,on=["PatientID"])

        # get data
        print(df_all.columns)
        df_data = df_all.loc[:,df3.columns.to_list()]
        df_data = df_data.drop(["PatientID"],axis=1)
        # get label
        df_label = df_all.loc[:,df_dnad.columns.to_list()]
    else:
        df_data = df3
        df_label = None
    #print(df3.head(5))
    return df_data, df_label



#############################################################################
#              handle table data from different file type
#############################################################################
class tableWorker:
    def __init__(self,loc:str) -> None:
        """
        tableWorker
        include file related functions for table data
        current support csv, xlsx, xls, Rdata
        """
        self.loc = loc
        self.file_type = Path(loc).suffix
        self.df = None

    def update_loc(self,loc:str):
        """
        update the loc of the table
        in:
            loc: str, the new loc of the table
        """
        self.loc = loc
        self.file_type = Path(loc).suffix

    def read_table(self,key:str=None) -> pd.DataFrame:
        """
        read a table from a csv xlsx xls Rdata file
        in:
            key: str,(optional) the key of the table(only for Rdata),if Rdata only include one object,key is None
        change:
            self.df: pd.DataFrame, the table read from the file
        """
        if self.file_type == ".csv":
            self.df = pd.read_csv(self.loc)
        elif self.file_type == ".xlsx" or self.file_type == ".xls":
            self.df =  pd.read_excel(self.loc)
        elif self.file_type == ".Rdata":
            import pyreadr
            result = pyreadr.read_r(self.loc)
            self.df = result[key]
        else:
            raise ValueError("unsupported file type")

    def write_table(self,df:pd.DataFrame=None,name:str=None) -> None:
        """
        write a table to a csv xlsx xls Rdata file
        in:
            df: pd.DataFrame, the table to write to the file
            key: str,(optional) the key of the table(only for Rdata),if Rdata only include one object,key is None
        """
        df = df if df is not None else self.df
        if self.file_type == ".csv":
            df.to_csv(self.loc)
        elif self.file_type == ".xlsx" or self.file_type == ".xls":
            df.to_excel(self.loc)
        elif self.file_type == ".Rdata":
            import pyreadr
            assert name is not None
            pyreadr.write_rdata(self.loc,df,df_name=name)
        elif self.file_type == "Rds":
            import pyreadr
            pyreadr.write_rds(self.loc,df)
        else:
            raise ValueError("unsupported file type")

    def show_csv_info(self,df:pd.DataFrame=None) -> None:
        """
        Get related info for a csv file
        in:
            df: pd.DataFrame, the table to get info
        """
        df = df if df is not None else self.df
        print("Keys of dataframe file is {}".format(df.keys()))
        print("There is {} items in csv file.".format(len(df)))
        print("The first 5 items in csv file is {}".format(df.head()))

    def df_to_list(self,keys:list,df:pd.DataFrame=None) -> list:
        """
        convert a dataframe to a list,each colume will be a list
        in:
            df: pd.DataFrame,(optional) the table to convert,if None,use self.df 
        out:
            list: list, the list of the dataframe
        """
        df = df if df is not None else self.df
        out = []
        for k in keys:
            out.append(df[k].values.tolist())
        return out
