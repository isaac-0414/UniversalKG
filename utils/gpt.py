from openai import OpenAI
import os
from time import time, sleep
import signal
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key = os.getenv("PROF_OPENAI_API_KEY"))

def gpt3_embedding(content, engine='text-embedding-ada-002') -> list[float]:
    """
    Wrapper of OpenAI text embedding call

    Parameters:
    content (str): the input document to be embedded
    engine (str): engine to use

    Returns:
    list[float]: the text embedding vector
    """
    response = client.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']
    return vector


def gpt_chat(messages: list[dict], model="gpt-3.5-turbo-1106", temperature=0.0, max_tokens=1024, stop=None, n=1, log=False) -> str:
    """
    Wrapper of OpenAI ChatCompletion call, allow 5 retries maximum. If no response after 60s, will cut the 
    connection and retry

    Parameters:
    messages (list[dict]): Input messages, e.g. [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    log (bool): whether or not to store the logs of GPT, default to be False

    Returns:
    str: the response from GPT
    """

    def handler(signum, frame):
        raise Exception('Function call timed out!')

    max_retry = 5
    retry = 0
    while True:
        # Set the signal handler
        signal.signal(signal.SIGALRM, handler)
        # Set a 50 second alarm
        signal.alarm(60)

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                n=n
            )
            gpt_response = completion.choices[0].message.content
            if log:
                filename = '%s_gpt.txt' % time()
                if not os.path.exists('gpt_logs'):
                    os.makedirs('gpt_logs')
                with open('gpt_logs/%s' % filename, 'w') as f:
                    f.write(str(messages) + '\n\n==========\n\n' + gpt_response)
            return gpt_response
        except Exception as oops:
            # handles timeout or error
            retry += 1
            if retry >= max_retry:
                # if used up the 5 retries, throw error
                return "GPT 3.5/4 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)
        finally:
            # Cancel the alarm
            signal.alarm(0)