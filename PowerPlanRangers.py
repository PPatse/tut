from openai import AzureOpenAI as AOpenAI
import chromadb
import pandas as pd
import pandasai
from pandasai.llm import AzureOpenAI as PAzureOpenAI
from pandasai import SmartDatalake, SmartDataframe, agent
import os
import chromadb
import re
import glob
import markdownify
import numpy as np
import scipy.stats as stats

chroma_path = "testchroma_db"
selected_collection = "default_collection"
# Create an empty DataFrame with specified columns
df = pd.DataFrame({'RAG Question': [], 'RAG Answer': [], 'Retrieved': [], 'Similarity': [],'Tokens': []})

#connections
azure_client = AOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY").strip('"'),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT").strip('"'),
)

pazure_llm = PAzureOpenAI(
    api_token = os.getenv("AZURE_OPENAI_API_KEY").strip('\"'),   
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT").strip('\"'),
    api_version="2024-06-01",
    deployment_name="gpt-4o"
)
# Functions
# Conan functions
def conanFAQ_test(question_file, result_file = "FAQ_result.csv", collection_name=selected_collection, chromapath=chroma_path):
    global df
    with open(question_file, "r", encoding="utf-8") as file:
        questions = file.readlines()
        file.close()
    for question in questions:
        conanFAQ(question, collection_name, chromapath)
    print("Test results saved to: " + result_file)
    df.to_csv(result_file, index=False)

def conanchat(user_query,collection_name=selected_collection, chromapath=chroma_path):
    retrieved_ragfiles = get_relevant_context(user_query, relpath = chromapath, relcollection = collection_name)
    given_text = organize_retrieved_content(retrieved_ragfiles)
    #similarity =organize_retrieved_content(retrieved_ragfiles, show_doc=False, show_meta=False, show_dist=True)
    conanresponse = azure_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context. The retrieved content is: "+ given_text},
                    {"role": "user", "content":user_query},
                    {"role": "assistant", "content": "You will clarify if the retrieved content was utilized to answer the user. Let the user know how helpful the given text was for answering the question and give it a score, which you will refer to as the RelevancyScore from 1 to 10. Put that score at the beginning of your answer."},
                    ],
                    temperature=0,      
                    )
    conananswer= conanresponse.choices[0].message.content.strip()
    return conananswer
    
def conanFAQ(user_query,collection_name=selected_collection, chromapath=chroma_path):
    global df  # Declare df as global
    retrieved_ragfiles = get_relevant_context(user_query, relpath = chromapath, relcollection = collection_name)
    given_text = organize_retrieved_content(retrieved_ragfiles)
    #similarity =organize_retrieved_content(retrieved_ragfiles, show_doc=False, show_meta=False, show_dist=True)
    similarity = distanceCalc(retrieved_ragfiles)
    conanresponse = azure_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context. The retrieved content is: "+ given_text},
                    {"role": "user", "content":user_query},
                    {"role": "assistant", "content": "You will clarify if the retrieved content was utilized to answer the user. Let the user know how helpful the given text was for answering the question and give it a score, which you will refer to as the RelevancyScore from 1 to 10. Put that score at the beginning of your answer."},
                    ],
                    temperature=0,      
                    )
    conananswer= conanresponse.choices[0].message.content.strip()
    conantokens = conanresponse.usage.total_tokens
    new_row = pd.DataFrame({"RAG Question": [user_query], "RAG Answer": [conananswer], "Retrieved": [given_text], "Similarity": [similarity],"Tokens": [conantokens]})
    print(new_row)
    df = pd.concat([df, new_row], ignore_index=True)
    
#Scotty functions

def scottyFAQ(source_file, rephasedFAQ ="scottyFAQ.txt"):
    with open(source_file, 'r', encoding="utf-8") as file:
    # Read the contents of the file
        content = file.read()
        print("reading " + source_file + " writing " + rephasedFAQ)
    scottyresponse = azure_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant that rewrites and reformats content as an FAQ"},
                    {"role": "user", "content":"Rewrite the following as a FAQ: " + content},
                    {"role": "assistant", "content": "For similar questions and answers, merge them into a single set of question and answer. You will remove out information about the RelevancyScore, the retrieved content, and given text."},
                    ],
                    temperature=0.2,      
                    )
    scottyanswer= scottyresponse.choices[0].message.content.strip()
    with open(rephasedFAQ, 'w', encoding="utf-8") as file:
        file.write(scottyanswer)
    return scottyanswer

def scottyEssay(source_file, topic, essay ="scottyessay.txt"):
    with open(source_file, 'r', encoding="utf-8") as file:
    # Read the contents of the file
        content = file.read()
        print("reading " + source_file + " writing " + essay)
    scottyresponse = azure_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant that tries to write short essay on this topic. You use given context, and will search out for more context. " + topic},
                    {"role": "user", "content":"This is some given context: " + content},
                    {"role": "assistant", "content": "You will organize this content like a highschool level essay."},
                    ],
                    temperature=0.6,      
                    )
    scottyanswer= scottyresponse.choices[0].message.content.strip()
    with open(essay, 'w') as file:
        file.write(scottyanswer)
    return scottyanswer

def scottyarticle(source_file, topic, article ="scottyarticle.txt"):
    with open(source_file, 'r', encoding="utf-8") as file:
    # Read the contents of the file
        content = file.read()
        print("reading " + source_file + " writing " + article)
    scottyresponse = azure_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant that tries to write an article on this topic: " + topic + " You use given context, and will search out for more context."},
                    {"role": "user", "content":"This is some given context: " + content},
                    {"role": "assistant", "content": "You will organize the article with an introduction, subheadings and analysis."},
                    ],
                    temperature=0.6,      
                    )
    scottyanswer= scottyresponse.choices[0].message.content.strip()
    with open(article, 'w') as file:
        file.write(scottyanswer)
    return scottyanswer

def scottyquiz(source_file, topic, quiz ="scottyquiz.txt", qnum=20):
    with open(source_file, 'r', encoding="utf-8") as file:
    # Read the contents of the file
        content = file.read()
        print("reading " + source_file + " writing " + quiz)
    scottyresponse = azure_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant that tries to write a multiple choice quiz based on this topic: " + topic + "and the given context."},
                    {"role": "user", "content":"This is some given context: " + content},
                    {"role": "assistant", "content": "Make a quiz of " + str(qnum) + "questions and create an answer key at the end. In the answer key explain the answer."},
                    ],
                    temperature=0.6,      
                    )
    scottyanswer= scottyresponse.choices[0].message.content.strip()
    with open(quiz, 'w') as file:
        file.write(scottyanswer)
    return scottyanswer

# Panda Functions

def load_pandas(source_folder= os.getcwd() ):
    dataframes =[]
    csv_files = glob.glob(os.path.join(source_folder, "*.csv"))
    print('Location:', source_folder)  
    for csv_file in csv_files:
        # Read the CSV file
        df = pd.read_csv(csv_file, encoding="utf-8")
        print('Loading File Name:', csv_file.split("\\")[-1]) 
        dataframes.append(df)
    return dataframes

def pandaq( source_folder = os.getcwd(), new_question_file = "PandaQuestions.txt", question_num=10):
    pandasai.clear_cache()
    smart_datalake = SmartDatalake(load_pandas(source_folder),config={"llm": pazure_llm})
        #pandaresponse = str(smart_datalake.chat("Create 10 new questions based on the RAG questions and RAG answers."))
    attempts = 0
    while True:
        try:
                pandaresponse = (smart_datalake.chat("Create " + str(question_num) + " completely new questions based on the RAG questions and RAG answers."))
                pandaresponse.to_csv(new_question_file, sep='\t', index=False)
                # Open the original file in read mode
                with open(new_question_file, 'r') as file:
                    lines = file.readlines()
                # Write all lines except the first one to a new file, this will get rid of the PandasAI adding the column name.
                with open(new_question_file, 'w') as file:
                    file.writelines(lines[1:])
                    print("New Questions written to "+ new_question_file)
                break
        except:
            attempts += 1
            print("Attempt number: " + str(attempts))
            if attempts == 15:
                break
def pandaFAQ (source_folder = os.getcwd(), FAQ_file = "FAQ.txt", filter= "the top 50 according to RelevancyScore."):
    smart_datalake = SmartDatalake(load_pandas(source_folder),config={"llm": pazure_llm})
    attempts = 0
    
    while True:
        try:
                pandaresponse = (smart_datalake.chat("Create a list of the RAG questions and RAG answers, only include "+ filter))
                pandaresponse.to_csv(FAQ_file, sep='\t', index=False)
                # Open the original file in read mode
                # Write all lines except the first one to a new file, this will get rid of the PandasAI adding the column name.
                # with open(FAQ_file, 'w') as file:
                #     file.write(pandaresponse)
                #     print("FAQ written to " + FAQ_file)
                break
        except Exception as e:
            attempts += 1
            print("Attempt number: " + str(attempts))
            if attempts == 10:
                break
            
def pandachat(user_query = None, source_folder = os.getcwd() ):
    smart_datalake = SmartDatalake(load_pandas(source_folder),config={"llm": pazure_llm})
    try:
        pandaresponse = str(smart_datalake.chat(user_query))
    except:
        pandaresponse = "sorry, the provided tables could not provide any answers."
    print("The answer is" + pandaresponse)
    return pandaresponse
# processing function

def markdowner(source_file):
    with open(source_file, "r", encoding="utf-8") as file:
        text = file.read()
        convertFlare = markdownify.markdownify(text, heading_style="ATX")
        # add some FlareSpecific stuff  
    return convertFlare

def markdownerfolder(source_folder, md_folder="markdowned"):
        # Check whether the specified path exists or not
    isExist = os.path.exists(md_folder)
    if not isExist:
        os.makedirs(md_folder)
        
    premd_files = [
        f
        for f in os.listdir(source_folder)
        if os.path.isfile(os.path.join(source_folder, f))
    ]
    for premd_file in premd_files:
        mdconvert = markdowner(source_folder + "\\" + premd_file)
        with open(md_folder + "\\" + premd_file, 'w', encoding="utf-8") as file:
            print("Markdown conversion of " + premd_file)
            file.write(mdconvert)
            file.close()

def chunker(source_file, source_section_pattern=None, chunk_folder="chunked", max_chunk_size=1000):
    with open(source_file, "r", encoding="utf-8") as file:
        text = file.read()
        print("chunking " + source_file)
        file.close()
    # Normalize whitespace and clean up text
    text = re.sub(r"\s+", " ", text).strip()
    # Split text into chunks by sentences, respecting a maximum chunk size and on spaces following sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?]) +", text)
    if source_section_pattern == None:
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            # Check if the current sentence plus the current chunk exceeds the limit
            if (
                len(current_chunk) + len(sentence) + 1 < max_chunk_size
            ):  # +1 for the space
                current_chunk += (sentence + " ").strip()
            else:
                # When the chunk exceeds 1000 characters, store it and start a new one
                chunks.append(current_chunk)
                current_chunk = sentence + " "
        if current_chunk:
            # Don't forget the last chunk!
            chunks.append(current_chunk)
    else:
        section_pattern = source_section_pattern
        chunks = re.split(section_pattern, text)
    # pull title as metadata
    with open(
        chunk_folder + "\\" + source_file.split("\\")[-1], "w", encoding="utf-8"
    ) as c_file:
        for chunk in chunks:
            # Write each chunk to its own line with two newlines to separate chunks
            c_file.write(chunk.strip() + "\n")

def folder_chunk_n_embed(source_folder, chunk_folder="chunked", collection_name="default_collection", chromapath="chroma_db", chunk_regex=None, max_chunk_size=1000):
    chroma_client = chromadb.PersistentClient(path=chromapath)
    collection = chroma_client.get_or_create_collection(
    name=collection_name, metadata={"hnsw:space": "cosine"}
    )
   
    prechunk_files = [
        f
        for f in os.listdir(source_folder)
        if os.path.isfile(os.path.join(source_folder, f))
    ]
   # Check whether the specified path exists or not
    isExist = os.path.exists(chunk_folder)
    if not isExist:
        os.makedirs(chunk_folder)
    

    for prechunk_file in prechunk_files:
        chunker(source_folder + "\\" + prechunk_file, chunk_regex, chunk_folder, max_chunk_size)

    chunked_files = [f for f in os.listdir(chunk_folder + "\\") if os.path.isfile(os.path.join(chunk_folder + "\\", f))]
    for chunked_file in chunked_files: 
         with open(chunk_folder + "\\" + chunked_file, "r", encoding="utf-8") as c_File:
            chunked_content = c_File.readlines()
            max_id = collection.count()
            for index, content in enumerate(chunked_content):
                collection.upsert(
                documents=[content],
                ids=[str(max_id + index + 1)],
                metadatas=[{"Source": chunked_file}],
            )
            print(str(index + 1) +" chunk(s) embedded from " + chunked_file + " into the collection: " + collection_name) 

def get_relevant_context(user_input, top_k=2, relpath=chroma_path, relcollection= selected_collection ):
    relchroma_client = chromadb.PersistentClient(path=relpath)
    # creating a collection
    relcollection = relchroma_client.get_or_create_collection(
    name=relcollection, metadata={"hnsw:space": "cosine"}
)
    relevant_context = relcollection.query(
        query_texts=[user_input],  # Chroma will embed this for you
        n_results=top_k,  # how many results to return
    )
    return relevant_context

def organize_retrieved_content(
    query_result, show_doc=True, show_meta=True, show_dist=False
):
    organized_data = []
    for i in range(len(query_result["documents"][0])):
        parts = []
        if show_doc:
            document = query_result["documents"][0][i]
            parts.append(f"Document: {document}")
        if show_meta:
            metadata = query_result["metadatas"][0][i]
            metadata_str = ", ".join(
                [f"{key}: {value}" for key, value in metadata.items()]
            )
            parts.append(f"Metadata: {metadata_str}")
        if show_dist:
            distance = query_result["distances"][0][i]
            parts.append(f"Distance: {distance}")
        organized_data.append("\n".join(parts))
    return "\n\n".join(organized_data)

def distanceCalc(query_result):
    distanceresponse =""
    for i in range(len(query_result["documents"][0])):
        distance = round(query_result["distances"][0][i], 3)
        distanceresponse = distanceresponse + str(distance) + ", "
    return (distanceresponse.rstrip(', '))

# reporting stat functions

def relscoreProbability(relscores, benchmark=5.5, confidence_level=0.95):
    # Convert scores to a NumPy array
    scores = np.array(relscores)
    
    # Calculate the mean and standard deviation of the relevancy scores
    mean = np.mean(scores)
    print("The mean relevancy score was: " + str(mean))
    std_dev = np.std(scores, ddof=1)
    print("The standard deviation was: " + str(std_dev))
    
    # Calculate the margin of error
    n = len(scores)
    print("Sample size " +str(n))
    if n > 30:
        z_critical = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_critical * (std_dev / np.sqrt(n))
    else:
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n - 1)
        margin_of_error = t_critical * (std_dev / np.sqrt(n))
    
    # Calculate the confidence interval
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    print(f"The {confidence_level*100:.1f}% confidence interval for the mean is: {confidence_interval}")
    
    # Calculate the probability of a score being higher than the benchmark
    if n > 30:
        z_score = (benchmark - mean) / std_dev
        probability = 1 - stats.norm.cdf(z_score)
    else:
        t_score = (benchmark - mean) / (std_dev / np.sqrt(n))
        probability = 1 - stats.t.cdf(t_score, df=n - 1)
    
    print(f"The probability of having a relevancy score higher than {benchmark} is {probability*100:.2f}%")
    return probability

def clean_file(input_file, output_file):
    try:
        # Read the file with error handling
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        # Write the cleaned content to a new file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"File cleaned and saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    #folder_chunk_n_embed(source_folder="WPAsource", chunk_folder="chunkWPA", collection_name="info_collection", chromapath="testchroma_db", chunk_regex=r"</h[23]>\s*", max_chunk_size=100)
    #conanFAQ_test(question_file= "QuestionforConan.txt", result_file = "conan_vault\ConanTestResults.csv", collection_name="info_collection", chromapath="testchroma_db")
    pass