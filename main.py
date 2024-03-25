import os
import pickle
import click

import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, util

from utilities.utils import *

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='debug mode flag', default=False)
@click.pass_context
def router_cmd(ctx: click.Context, debug):
    ctx.obj['debug_mode'] = debug 
    invoked_subcommand = ctx.invoked_subcommand 
    if invoked_subcommand is None:
        logger.debug('no subcommand were called')
    else:
        logger.debug(f'{invoked_subcommand} was called')
 
@router_cmd.command()
@click.option('--path2data', help='', type=str, default='data/')
def grabber(path2data):
    logger.debug('Loading data...')
    if not os.path.exists(path2data):
        os.makedirs(path2data)
    data = load_dataset('ashraq/fashion-product-images-small', split='train')
    with open(os.path.join(path2data, 'fashion_data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
       
@router_cmd.command()
@click.option('--path2data', help='path to data', type=click.Path(True), default='data/')
def vectorizer(path2data):
    logger.debug('Vectorizing...')
    with open(os.path.join(path2data, 'fashion_data.pkl'), 'rb') as f:
        fashion_data:Dataset = pickle.load(f)
    print(fashion_data.features)
    
    fashion_images = fashion_data['image']
    fashion_data = fashion_data.remove_columns('image')
    fashion_df = fashion_data.to_pandas()
    print(fashion_df.head())
    model = SentenceTransformer('clip-ViT-B-32')
    embeddings = model.encode([image for image in fashion_images])
    
    with open(os.path.join(path2data, 'fashion_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)


@router_cmd.command()
@click.option('--path2data', help='path to data', type=click.Path(True), default='data/')
def search(path2data):
    logger.debug('Semantic searching...')
    
    model = SentenceTransformer('clip-ViT-B-32')
    with open(os.path.join(path2data, 'fashion_embeddings.pkl'), 'rb') as f:
        embeddings = pickle.load(f)
    
    with open(os.path.join(path2data, 'fashion_data.pkl'), 'rb') as f:
        fashion_data:Dataset = pickle.load(f)
    
    fashion_images = fashion_data['image']
    
    query = input('Query: ')
    query_embedding = model.encode(query)
    results = util.semantic_search(query_embedding, embeddings, top_k=15)[0]
    
    _, axes = plt.subplots(3, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fashion_images[results[i]['corpus_id']])
        ax.axis('off')
    
    plt.show()

if __name__ == '__main__':
    router_cmd(obj={})

if __name__ == '__main__':
    logger.info('Processing...')