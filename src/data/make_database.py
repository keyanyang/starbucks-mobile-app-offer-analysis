# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import sqlalchemy as sa
import os


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs scripts to turn raw data from (../raw) into
    a queryable/indexable form (an SQL database saved in ../raw).

    Parameters
    ----------
    input_filepath: str
        The file location of the input data.
    output_filepath: str
        The file location of the output SQLite database.

    Returns
    -------
    None
    """
    logger = logging.getLogger(__name__)
    logger.info('making database from raw data')

    raw_data_files = [filename for filename in os.listdir(input_filepath)
                      if filename.endswith('json')]

    db_path = os.path.join(output_filepath, 'raw_data.sqlite')
    e = sa.create_engine('sqlite:///' + db_path)
    for raw_data_file in sorted(raw_data_files):
        df = pd.read_json(f'{input_filepath}/{raw_data_file}',
                    orient='records', lines=True)
        for col in list(df):
            if df[col].dtype == object:
                df[col] = df[col].astype('str')

        table_name = raw_data_file.split('.')[0]
        df.to_sql(table_name, e, index=False)
        logger.info(f'table {table_name} created')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
