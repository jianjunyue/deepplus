from pandasql import sqldf
# pysqldf = lambda q: sqldf(q, globals())
class SQLHelper:
    def __init__(self):
        return

    def select(sql):
        """
        sql查询
        :param sql:
        :return:
        """
        table = sqldf(sql, locals())
        return table

    def select(user_profile,sql):
        """
        sql查询
        :param sql:
        :return:
        """
        table = sqldf(sql, locals())
        return table