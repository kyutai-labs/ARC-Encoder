# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "argparse",
#     "dataclasses",
#     "pathlib",
#     "setuptools",
#     "vllm",
#     "numpy",
# ]
# ///
import argparse
import json
from pathlib import Path
from vllm import LLM, SamplingParams  # type: ignore
from dataclasses import dataclass
import numpy as np
import random
import re  # noqa: F401


ATLAS_PATH = "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Atlas/enwiki-dec2021/text-list-100-sec.jsonl"

# HOTPOT_EXAMPLE = 'Document: For Against is an American post-punk/dream pop band from Lincoln, Nebraska, United States. Despite numerous lineup shuffles and some periods of dormancy, the band has produced material steadily since 1984. \n Local H is an American rock band originally formed by guitarist and vocalist Scott Lucas, bassist Matt Garcia, drummer Joe Daniels, and lead guitarist John Sparkman in Zion, Illinois in 1990. \nQuestion: Are Local H and For Against both from the United States?\nAnswer: yes\n\nDocument: The Androscoggin Bank Colis\u00e9e (formerly Central Maine Youth Center, Central Maine Civic Center and Lewiston Colisee) is a 4,000 capacity (3,677 seated) multi-purpose arena, in Lewiston, Maine, that opened in 1958. The Androscoggin Bank Colis\u00e9e was built to replace St. Dominics Regional High School Arena, and initially constructed and operated by the Catholic parish of SS. Peter and Paul. Currently, it is the home of the Maine Nordiques of the North American Hockey League. The Colisee is also used for concerts, conventions and trade shows. There is 17,000 square feet (1600 m2) of exhibit space. For conventions, the Colisee can accommodate up to 4,800 patrons. \nQuestion: The arena where the Lewiston Maineiacs played their home games can seat how many people?\nAnswer: 3,677 seated\n\nDocument: Thinking Fellers Union Local 282 were an American experimental indie rock group, which was formed in 1986 in San Francisco, California, United States, though half of its members are from Iowa. Their albums combine lo-fi noise rock and ambient sounds (referred to as "Feller filler") with tightly constructed rock and pop songs. The band has a small but intensely loyal cult following. Band members are Brian Hageman, Mark Davies, Anne Eickelberg, Hugh Swarts and Jay Paget. Hageman was also a member of the Iowa City based group, Horny Genius. The band achieved their greatest critical and commercial success in the mid-1990s, when they signed with the indie rock label Matador Records. It was during this time that Thinking Fellers produced their most prominent albums. \nDig is an American alternative rock band from Los Angeles, California. Formed in 1991, they achieved success with their 1993 album Dig, which featured the charting single "Believe".\nQuestion: Were both of the following rock groups formed in California: Dig and Thinking Fellers Union Local 282?\nAnswer: yes\n\n'  # noqa: E501
# NQ_EXAMPLE = 'Document: The South Lawn at the White House in Washington, D.C., is directly south of the house and is bordered on the east by East Executive Drive and the Treasury Building, on the west by West Executive Drive and the Old Executive Office Building, and along its curved southern perimeter by South Executive Drive and a large circular public lawn called The Ellipse. Since the address of the White House is 1600 Pennsylvania Avenue NW, and the North Lawn faces Pennsylvania Avenue, the South Lawn is sometimes described as the back lawn of the White House.\nQuestion: Which side of the white house is the front?\nAnswer: North\n\nDocument: Dame Flora Louise Shaw, Lady Lugard (born 19 December 1852 \u2013 25 January 1929), was a British journalist and writer. She is credited with having coined the name "Nigeria".\nQuestion: nigeria was given it\'s name by who?\nAnswer: Flora Louise Shaw\n\nDocument: Anglo-American Peace Centenary (1814\u20131914) was in 1914 to celebrate the lasting peace between Britain and the United States. They last fought in the War of 1812, and those hostilities were ended formally on December 24, 1814, with the signing of the Treaty of Ghent.\nQuestion: when did the united states and britain sign a peace treaty\nAnswer: 1814\n\n'  # noqa: E501
# TRIVIA_EXAMPLE = "Document: Super Bowl XX was an American football game between the National Football Conference (NFC) champion Chicago Bears and the American Football Conference (AFC) champion New England Patriots to decide the National Football League (NFL) champion for the 1985 season. The Bears defeated the Patriots by the score of 46\u201310, capturing their first NFL championship since 1963, three years prior to the birth of the Super Bowl. Super Bowl XX was played on January 26, 1986 at the Louisiana Superdome in New Orleans. This was the fourth Super Bowl and, to date, the last time in which both teams made their Super Bowl debuts. The Bears entered the game after becoming the second team in NFL history to win\nQuestion: Who won Super Bowl XX?\nAnswer: Chicago bears\n\nDocument: 1950 \u2013 Senator Joseph McCarthy gains power, and McCarthyism (1950\u20131954) begins ; 1950 \u2013 McCarran Internal Security Act ; 1950 \u2013 Korean War begins ; 1950 \u2013 The comic strip Peanuts, by Charles M. Schulz, is first published ; 1950 \u2013 NBC airs Broadway Open House a late-night comedy, variety, talk show through 1951. Hosted by Morey Amsterdam and Jerry Lester and Dagmar, it serves as the prototype for The Tonight Show ; 1950 \u2013 Failed assassination attempt by two Puerto Rican nationals on President Harry S. Truman while the President was living at Blair House. ; 1951 \u2013 22nd Amendment, establishing term\nQuestion: Who was President when the first Peanuts cartoon was published?\nAnswer: 33rd President of the United States\n\nDocument: (Note: Original spelling variations left intact.) Humpty Dumpty sate on a wall, Humpti Dumpti had a great fall; Threescore men and threescore more, Cannot place Humpty dumpty as he was before. In 1842, James Orchard Halliwell published a collected version as: Humpty Dumpty lay in a beck. With all his sinews around his neck; Forty Doctors and forty wrights Couldn't put Humpty Dumpty to rights! The modern-day version of this nursery rhyme, as known throughout the UK since at least the mid-twentieth century, is as follows: Humpty Dumpty sat on a wall, Humpty Dumpty had a great fall; All the King's horses And all the King's men, Couldn't put Humpty together again. According to the Oxford English Dictionary, in the\nQuestion: According to the nursery rhyme, who couldnt put humpty back together again?\nAnswer: All the kings horses and all the kings men\n\n"  # noqa: E501
# SQUAD_EXAMPLE = 'Document: A self-described "modern-day feminist", Beyonc\u00e9 creates songs that are often characterized by themes of love, relationships, and monogamy, as well as female sexuality and empowerment. On stage, her dynamic, highly choreographed performances have led to critics hailing her as one of the best entertainers in contemporary popular music. Throughout a career spanning 19 years, she has sold over 118 million records as a solo artist, and a further 60 million with Destiny\'s Child, making her one of the best-selling music artists of all time. She has won 20 Grammy Awards and is the most nominated woman in the award\'s history. The Recording Industry Association of America recognized her as the Top Certified Artist in America during the 2000s decade. In 2009, Billboard named her the Top Radio Songs Artist of the Decade, the Top Female Artist of the 2000s and their Artist of the Millennium in 2011. Time listed her among the 100 most influential people in the world in 2013 and 2014. Forbes magazine also listed her as the most powerful female musician of 2015\nQuestion: In her music, what are some recurring elements in them?\nAnswer: love, relationships, and monogamy\n\nDocument: Following the disbandment of Destiny\'s Child in June 2005, she released her second solo album, B\'Day (2006), which contained hits "D\u00e9j\u00e0 Vu", "Irreplaceable", and "Beautiful Liar". Beyonc\u00e9 also ventured into acting, with a Golden Globe-nominated performance in Dreamgirls (2006), and starring roles in The Pink Panther (2006) and Obsessed (2009). Her marriage to rapper Jay Z and portrayal of Etta James in Cadillac Records (2008) influenced her third album, I Am... Sasha Fierce (2008), which saw the birth of her alter-ego Sasha Fierce and earned a record-setting six Grammy Awards in 2010, including Song of the Year for "Single Ladies (Put a Ring on It)". Beyonc\u00e9 took a hiatus from music in 2010 and took over management of her career; her fourth album 4 (2011) was subsequently mellower in tone, exploring 1970s funk, 1980s pop, and 1990s soul. Her critically acclaimed fifth studio album, Beyonc\u00e9 (2013), was distinguished from previous releases by its experimental production and exploration of darker themes.\nQuestion: What was the name of Beyonc\u00e9\'s second solo album?\nAnswer: B\'Day\n\nDocument: The main international airport serving Kathmandu and thus Nepal is the Tribhuvan International Airport, located about six kilometers (6 km (3.7 mi)) from the city centre. Operated by the Civil Aviation Authority of Nepal it has two terminals, one domestic and one international. At present, about 22 international airlines connect Nepal to other destinations in Europe, Asia and the Middle East, to cities such as Istanbul, Delhi, Kolkata, Singapore, Bangkok, Kuala Lumpur, Dhaka, Islamabad, Paro, Lhasa, Chengdu, and Guangzhou. A recent extension to the international terminal has made the distance to the airplanes shorter and in October 2009 it became possible to fly directly to Kathmandu from Amsterdam with Arkefly. Since 2013, Turkish Airlines connects Istanbul to Kathmandu. Regionally, several Nepali airlines operate from the city, including Agni Air, Buddha Air, Cosmic Air, Nepal Airlines and Yeti Airlines, to other major towns across Nepal.\nQuestion: From what city does Arkefly offer nonstop flights to Kathmandu?\nAnswer: Amsterdam\n\n'  # noqa: E501
# DONT_KNOW = 'Document: For example, a horse eats grass: the horse changes the grass into itself; the grass as such does not persist in the horse, but some aspect of it\u2014its matter\u2014does. The matter is not specifically described (e.g., as atoms), but consists of whatever persists in the change of substance from grass to horse. Matter in this understanding does not exist independently (i.e., as a substance), but exists interdependently (i.e., as a "principle") with form and only insofar as it underlies change. It can be helpful to conceive of the relationship of matter and form as very similar to that between parts and whole. For Aristotle, matter as such can only receive actuality from form; it has no activity or actuality in itself, similar to the way that parts as such only have their existence in a whole (otherwise they would be independent wholes).\nQuestion: Who said matter had actuality in and of itself?\nAnswer: I don\'t know.\n\nDocument: These quarks and leptons interact through four fundamental forces: gravity, electromagnetism, weak interactions, and strong interactions. The Standard Model of particle physics is currently the best explanation for all of physics, but despite decades of efforts, gravity cannot yet be accounted for at the quantum level; it is only described by classical physics (see quantum gravity and graviton). Interactions between quarks and leptons are the result of an exchange of force-carrying particles (such as photons) between quarks and leptons. The force-carrying particles are not themselves building blocks. As one consequence, mass and energy (which cannot be created or destroyed) cannot always be related to matter (which can be created out of non-matter particles such as photons, or even out of pure energy, such as kinetic energy). Force carriers are usually not considered matter: the carriers of the electric force (photons) possess energy (see Planck relation) and the carriers of the weak force (W and Z bosons) are massive, but neither are considered matter either. However, while these particles are not considered matter, they do contribute to the total mass of atoms, subatomic particles, and all systems that contain them.\nQuestion: What relation explains the carriers of the electric force?\nAnswer: I don\'t know.\n\nDocument: The term "matter" is used throughout physics in a bewildering variety of contexts: for example, one refers to "condensed matter physics", "elementary matter", "partonic" matter, "dark" matter, "anti"-matter, "strange" matter, and "nuclear" matter. In discussions of matter and antimatter, normal matter has been referred to by Alfv\u00e9n as koinomatter (Gk. common matter). It is fair to say that in physics, there is no broad consensus as to a general definition of matter, and the term "matter" usually is used in conjunction with a specifying modifier.\nQuestion: Physics has broadly agreed on the definition of what?\nAnswer: I don\'t know.' # noqa: E501

# instruction_prompts_wdoc = [
#     'You\'re given several examples of QA pairs based on different documents. For each example, the format is: "Document: <document>\nQuestion: <question>\nAnswer: <answer>".\nHere are a few examples:\n {examples}Now, based on the following document and previous examples, generate one question and its corresponding answer, in the same format.\nDocument: {passages}',
#     'Given a document, generate a question that can be answered directly from the text, and provide the answer. Use the following format: "Document: <document>\nQuestion: <question>\nAnswer: <answer>".\nFollow these examples:\n {examples}Now try with this new document:\n {passages}',
#     "Below are examples of how to synthesize QA pairs from a given document. Follow the same pattern to generate one question and answer for the new document provided.\n{examples}Now do the same for this document:\n {passages}",
# ]

# instruction_prompts_wodoc = [
#     "Generate a short paragraph that contains some factual or descriptive information. Then write a question that can be answered from the paragraph, and provide the answer. Use the same format as this following examples: {examples}",
#     "Write a 2–4 sentence document that includes specific information. Then, formulate a question that can be answered from this document, and give the correct answer. Present everything in these examples: {examples}",
#     "Create a self-contained QA example. First, generate a brief passage with a clear, factual claim or detail. Then, write a question whose answer is explicitly stated in the passage. Finish with the correct answer. Follow this format: {examples}Make sure the passage naturally contains the answer.",
# ]

instruction_prompts = {
    # "QA1": "Generate a question and its answer based solely on the content of the document above.",
    # "QA2": "Using the information in the document, create a factual QA pair.",
    # "QA3": "Write one question that can be answered from the previous passage, and provide the correct answer.",
    # "QA4": "Ask questions concerning the preceding passage and provide a short answer.",
    # "QA5": "Formulate a factual question. The question should require a short answer. Then, provide the answer.",
    "S1": "Summarize the key points of the document in 2 to 3 sentences.",
    "S2": "Provide a concise summary that captures the essence of the text above.",
    "S3": "Write a short summary of the previous document, focusing on its main message.",
    "T1": "Translate the previous document into Spanish.",
    "T2": "Render the document into fluent German while preserving its meaning.",
    "T3": "Provide a French translation of the text above.",
    "P1": "Paraphrase the document using clearer and simpler wording.",
    "P2": "Rewrite the passage based on its content without directly copying any phrasing.",
    "P3": "Rephrase the document in a style suitable for a younger audience.",
    "P4": "Reword the passage to make it more accessible while keeping the original meaning.",
    # "R1": "Reconstruct perfectly the passage.",
}

translate_prompts = [
    "Translate the previous document into {language}.",
    "Render the document into fluent {language} while preserving its meaning.",
    "Provide a {language} translation of the text above.",
    "As a translator, convert the document into {language} while maintaining its original meaning.",
    "Translate the document into {language} while ensuring clarity and accuracy.",
]

LANGUAGES = ["Spanish", "French", "German", "Danish"]


def mixture_dataset(
    prop_ds: list[float],
    dataset_paths: list[str],
    max_samples: int,
    output_path: str,
    seed: int = 28,
) -> list[str]:
    """
    Create a mixture of datasets based on the given probabilities.
    """
    random.seed(seed)
    n_sample_per_datasets = prop_ds * max_samples / np.sum(prop_ds)
    new_ds = []
    for i, ds_path in enumerate(dataset_paths):
        ds_data = []
        with open(ds_path, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(reversed(lines)):
                dataset_list = []
                if idx == n_sample_per_datasets[i]:
                    break
                sample = json.loads(line)
                ds_data.append(sample)
        new_ds.extend(random.sample(ds_data, n_sample_per_datasets[i]))
    with open(output_path, "w") as f:
        for item in new_ds:
            f.write(json.dumps(item) + "\n")
    return dataset_list


def passage_filter(passage: str, min_alpha_ratio: float = 0.75) -> bool:
    """
    Filter the passage to remove unwanted characters and format it.
    """
    if len(passage) < 10 or len(passage) > 16000:
        return False
    alpha_count = sum(c.isalpha() for c in passage)
    return alpha_count / len(passage) >= min_alpha_ratio


@dataclass
class Batch:
    passage: str
    instruction_prompts: str
    prompt_key: str


def dataset_from_file(file_path, n_passages: int = 1):
    sample = []
    n_sample = random.randint(1, n_passages)
    while True:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(reversed(lines)):
                data = json.loads(line)
                if passage_filter(data["text"]):
                    sample.append(data["text"].strip())
                if len(sample) == n_sample:
                    yield ("\n".join(sample))[:16000]
                    n_sample = random.randint(1, n_passages)
                    sample = []


def dataloader_from_file(
    file_path,
    batch_size,
    n_passages: int = 1,
    translate: bool = False,
):
    dataset = dataset_from_file(file_path, n_passages)

    probs = []
    for key in list(instruction_prompts.keys()):
        if key != "R1":
            probs.append(1)
        else:
            probs.append(0.1)
    probs = probs / np.sum(probs)
    batch_list = []
    while True:
        if translate:
            language = random.choice(LANGUAGES)
            prompt = random.choice(translate_prompts).format(
                language=language
            )
            prompt_key = language
        else:
            prompt_key = np.random.choice(list(instruction_prompts.keys()), p=probs)
            prompt = instruction_prompts[prompt_key]

        passage = next(dataset)
        batch_list.append(
            Batch(
                instruction_prompts=prompt, passage=passage, prompt_key=str(prompt_key)
            )
        )
        if len(batch_list) == batch_size:
            yield batch_list
            batch_list = []


def synthesize_data(
    temperature: float,
    top_p: float,
    model_folder_path: str,
    batch_size: int,
    data_path: str,
    output_path: str,
    max_steps: int,
    download_freq: int,
    ds_name: str,
    translate: bool = False,
    n_passages: int = 1,
):
    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True, exist_ok=True)
    out_file_path = output_path + ds_name + model_folder_path.split("/")[-1] + ".jsonl"

    llm = LLM(model=model_folder_path, dtype="bfloat16", max_model_len=16384)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=1024,
    )
    dataloader = dataloader_from_file(data_path, batch_size, n_passages, translate)
    output_buffer = []
    n_samples = 0
    for step in range(max_steps):
        batch = next(dataloader)
        n_samples += len(batch)
        output_buffer.extend(
            [
                {
                    "question": b.instruction_prompts,
                    "passage": b.passage,
                    "answer": b.passage,
                    "full_output": b.passage,
                    "prompt_key": "R1",
                }
                for b in batch
                if b.prompt_key == "R1"
            ]
        )

        batch = [
            b for b in batch if b.prompt_key != "R1"
        ]  # Remove the batch with prompt_key "R1"

        text_prompts = [
            "Answer to the instructions without any additional comments!\n\nDocument: "
            + b.passage
            + "\n\n"
            + b.instruction_prompts
            for b in batch
        ]

        outputs = llm.generate(text_prompts, sampling_params)
        for i, output in enumerate(outputs):
            if output.finished:
                question, answer = (
                    batch[i].instruction_prompts,
                    output.outputs[0].text.strip(),
                )
                # question, answer = reformat_example(
                #     text=output.outputs[0].text.strip(), prompt_key=batch[i].prompt_key
                # )
                output_buffer.append(
                    {
                        "question": question,
                        "passage": batch[i].passage.replace("\nDocument: ", "\n\n"),
                        "answer": answer,
                        "full_output": output.outputs[0].text.strip(),
                        "prompt_key": batch[i].prompt_key,
                    }
                )

        if len(output_buffer) >= download_freq:
            print(
                "Current step:",
                step,
                "N SAMPLES:",
                n_samples,
                "Example:",
                output_buffer[-1],
            )
            with open(out_file_path, "a") as f:
                for item in output_buffer:
                    f.write(json.dumps(item) + "\n")
            output_buffer = []


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer",
        type=str,
        default="google/gemma-3-27b-it",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=ATLAS_PATH,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/synthesized/",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000000,
    )

    parser.add_argument(
        "--download_freq",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--translate",
        action="store_true",
    )

    parser.add_argument(
        "--n_passages",
        default=1,
        type=int,
    )

    parser.add_argument("--ds_name", type=str, default="synth")

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    synthesize_data(
        temperature=args.temperature,
        top_p=args.top_p,
        model_folder_path=args.transformer,
        batch_size=args.batch_size,
        data_path=args.data_path,
        output_path=args.output_path,
        max_steps=args.max_steps,
        download_freq=args.download_freq,
        ds_name=args.ds_name,
        translate=args.translate,
        n_passages=args.n_passages,
    )
