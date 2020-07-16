import tensorflow as tf
from tensorflow.python.client import device_lib

import pandas as pd


device_name = tf.test.gpu_device_name()
device_name #check gpu 


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
get_available_gpus()

train_f = './paraphrase_corpus/msr_paraphrase_train.txt'
test_f = './paraphrase_corpus/msr_paraphrase_test.txt'

traindf = pd.read_csv(train_f, sep='\t', encoding='utf-8', error_bad_lines=False)
testdf = pd.read_csv(test_f, sep='\t', encoding='utf-8', error_bad_lines=False)

max_length = 512
batch_size = 6

from transformers import AutoTokenizer, TFBertForSequenceClassification

model_name="bert-base-uncased" 

tokenizer = AutoTokenizer.from_pretrained(model_name)

def convert_example_to_feature(s1, s2):
  
  # combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length
  
  return tokenizer.encode_plus(s1, s2, 
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = max_length, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
                truncation=True
                )

# map to the expected input to TFBertForSequenceClassification, see here 
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds, limit=-1):

  # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    if (limit > 0):
        ds = ds.take(limit)
    
    for i, r in ds.iterrows():
        s1 = str(r['#1 String']) if len(str(r['#1 String'])) else ' '
        s2 = str(r['#2 String']) if len(str(r['#2 String'])) else ' '
        label = int(r['Quality'])
        bert_input = convert_example_to_feature(s1, s2)
  
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])

    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

ds_train_encoded = encode_examples(traindf).shuffle(10000).batch(batch_size)
ds_test_encoded = encode_examples(testdf).shuffle(10000).batch(batch_size)

### Model learning parameters
learning_rate=2e-5
number_of_epochs=1

model = TFBertForSequenceClassification.from_pretrained(model_name)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)

df1 = pd.read_csv('./reflections_collections/annotated_manual_primer_responses.csv', index_col=0)
df2 = pd.read_csv('./reflections_collections/full_median_length_primers.csv', index_col=0)
df3 = pd.read_csv('./reflections_collections/numshot_analysis_sample_coded.csv', index_col=0)
df4 = pd.read_csv('./reflections_collections/verified_gpt_reflections.csv', index_col=0)
df5 = pd.read_csv('./reflections_collections/verified_gpt_reflections_3shot.csv', index_col=0)

df1=df1[['prompt', 'response', 'reflection_gpt', 'gpt_valid_reflection']]
df2=df2[['prompt', 'response', 'reflection_gpt', 'gpt_valid_reflection']]
df3=df3[['prompt', 'response', 'reflection_gpt', 'gpt_valid_reflection']]
df4=df4[['prompt', 'response', 'reflection_gpt', 'gpt_valid_reflection']]
df5=df5[['prompt', 'response', 'reflection_gpt', 'gpt_valid_reflection']]

compatibledf = pd.DataFrame(columns=['#1 String', '#2 String', 'Quality'])

fulldf = pd.concat([df1,df2,df3,df4,df5] ,ignore_index=True)

compatibledf['#1 String'] = fulldf['prompt'] + ' ' + fulldf['response']
compatibledf['#2 String'] = fulldf['reflection_gpt']
compatibledf['Quality'] = fulldf['gpt_valid_reflection']

# create train/test split
import numpy as np
msk = np.random.rand(len(compatibledf)) < 0.8
finetune_train = compatibledf[msk]
finetune_test = compatibledf[~msk]
print(f"training size: {len(finetune_train)} test size: {len(finetune_test)}")

finetune_train_encoded = encode_examples(finetune_train).shuffle(10000).batch(batch_size)
finetune_test_encoded = encode_examples(finetune_test).shuffle(10000).batch(batch_size)

num_epochs = 7

finetuned_bert_history = model.fit(finetune_train_encoded, epochs=num_epochs, validation_data=finetune_test_encoded)

model.save("finetuned_bert")

fname = './unlabeled_reflections/four_and_five_shot.csv'

unlabeled = pd.read_csv(fname, index_col=0)

fourshot = unlabeled[unlabeled['num_shot']==4]
fiveshot = unlabeled[unlabeled['num_shot']==5]
sixshot = unlabeled[unlabeled['num_shot']==6]

fourshotinput = pd.DataFrame(columns=['#1 String', '#2 String', 'Quality', 'Probability'])
fourshotinput['#1 String'] = fourshot['prompt'] + ' ' + fourshot['response']
fourshotinput['#2 String'] = fourshot['reflection_gpt']
fourshotinput['Quality'] = -1
fourshotinput['Probability'] = -1

fiveshotinput = pd.DataFrame(columns=['#1 String', '#2 String', 'Quality', 'Probability'])
fiveshotinput['#1 String'] = fiveshot['prompt'] + ' ' + fiveshot['response']
fiveshotinput['#2 String'] = fiveshot['reflection_gpt']
fiveshotinput['Quality'] = -1
fiveshotinput['Probability'] = -1

sixshotinput = pd.DataFrame(columns=['#1 String', '#2 String', 'Quality', 'Probability'])
sixshotinput['#1 String'] = sixshot['prompt'] + ' ' + sixshot['response']
sixshotinput['#2 String'] = sixshot['reflection_gpt']
sixshotinput['Quality'] = -1
fiveshotinput['Probability'] = -1

# map to the expected input to TFBertForSequenceClassification, see here 
def map_example_to_dict(input_ids, attention_masks, token_type_ids):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }

def encode_test_examples(ds, limit=-1):

  # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []

    if (limit > 0):
        ds = ds.take(limit)
    
    for i, r in ds.iterrows():
        s1 = str(r['#1 String']) if len(str(r['#1 String'])) else ' '
        s2 = str(r['#2 String']) if len(str(r['#2 String'])) else ' '
        label = int(r['Quality'])
        bert_input = convert_example_to_feature(s1, s2)
  
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])

    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list)).map(map_example_to_dict)

# four shot
batch_size=6
topredict4shot = encode_test_examples(fourshotinput).batch(batch_size)

fourpredictions = model.predict(topredict4shot, verbose=1)
fourpreds = tf.nn.softmax(fourpredictions)
fourpreds_argmax =  tf.math.argmax(fourpreds, 1)
labels4shot = preds_argmax.numpy()

fourshotinput['Probability'] = fourpreds.numpy()
fourshotinput['Quality'] = labels4shot
fourshotreflectionaccuracy = sum(labels4shot)/len(labels4shot)
fourshotinput.to_csv('fourshot_autolabeled.csv')

# five shot
batch_size=6
topredict5shot = encode_test_examples(fiveshotinput).batch(batch_size)

fivepredictions = model.predict(topredict5shot, verbose=1)
fivepreds = tf.nn.softmax(fivepredictions)
fivepreds_argmax =  tf.math.argmax(fivepreds, 1)
labels5shot = fivepreds_argmax.numpy()

fiveshotinput['Probability'] = fivepreds.numpy()
fiveshotinput['Quality'] = labels5shot
fiveshotreflectionaccuracy = sum(labels5shot)/len(labels5shot)
fiveshotinput.to_csv('fiveshot_autolabeled.csv')

# six shot
batch_size=6
topredict6shot = encode_test_examples(sixshotinput).batch(batch_size)

sixpredictions = model.predict(topredict6shot, verbose=1)
sixpreds = tf.nn.softmax(sixpredictions)
sixpreds_argmax =  tf.math.argmax(sixpreds, 1)
labels6shot = sixpreds_argmax.numpy()

sixshotinput['Probability'] = sixpreds.numpy()
sixshotinput['Quality'] = labels6shot
sixshotreflectionaccuracy = sum(labels6shot)/len(labels6shot)
sixshotinput.to_csv('sixshot_autolabeled.csv')

dfa = pd.read_csv('./reflections_collections/annotated_manual_primer_responses.csv', index_col=0)
dfb = pd.read_csv('./reflections_collections/full_median_length_primers.csv', index_col=0)
dfc = pd.read_csv('./reflections_collections/numshot_analysis_sample_coded.csv', index_col=0)
dfd = pd.read_csv('./reflections_collections/verified_gpt_reflections.csv', index_col=0)
dfe = pd.read_csv('./reflections_collections/verified_gpt_reflections_3shot.csv', index_col=0)

dfa4 = dfa[dfa['num_shot']==4]
dfa5 = dfa[dfa['num_shot']==5]
dfa6 = dfa[dfa['num_shot']==6]
dfb4 = dfb[dfb['num_shot']==4]
dfb5 = dfb[dfb['num_shot']==5]
dfb6 = dfb[dfb['num_shot']==6]
dfc4 = dfc[dfc['num_shot']==4]
dfc5 = dfc[dfc['num_shot']==5]
dfc6 = dfc[dfc['num_shot']==6]

df4shot = pd.concat([dfa4,dfb4,dfc4],ignore_index=True)
df5shot = pd.concat([dfa5,dfb5,dfc5],ignore_index=True)
df6shot = pd.concat([dfa6,dfb6,dfc6],ignore_index=True)

print(f"4 shot: {df4shot.shape} 5 shot: {df5shot.shape} 6 shot: {df6shot.shape}")

fourshotmanualacc = sum(df4shot['gpt_valid_reflection'])/len(df4shot['gpt_valid_reflection'])
fiveshotmanualacc = sum(df5shot['gpt_valid_reflection'])/len(df5shot['gpt_valid_reflection'])
sixshotmanualacc = sum(df6shot['gpt_valid_reflection'])/len(df6shot['gpt_valid_reflection'])

print(f"4 shot: {fourshotmanualacc} 5 shot: {fiveshotmanualacc} 6 shot: {sixshotmanualacc}")