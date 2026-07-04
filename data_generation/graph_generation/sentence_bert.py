import argparse
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from common.util import read_line


class pipeline_runner:
    def __init__(self, args):
        self.args = args

        with open(args.input_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.similarity_threshold = args.similarity_threshold
        self.split_by_tool = args.split_by_tool
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def compute_scores(self):
        data = self.data

        input_sentences = []
        output_sentences = []
        for item in data:
            output_sentences.append(item['output'])
            input_sentences.append(item.get('input_params', []))

        output_embeddings = self.model.encode(output_sentences, convert_to_tensor=True)

        result = []
        # compute cosine similarity score for each output and each input parameter
        for i, inputs in enumerate(input_sentences):
            if not inputs:
                continue

            descriptions = [param["description"] for param in inputs]

            input_embeddings = self.model.encode(descriptions, convert_to_tensor=True)

            cosine_scores = util.pytorch_cos_sim(input_embeddings, output_embeddings)

            for k in range(len(inputs)):
                for j in range(len(output_sentences)):
                    sim_dict = {
                        'out_id': data[j]['docid'],  # document id of API j
                        'in_id': data[i]['docid'],  # document id of API i
                        'in_param': k,  # index of parameter of API i
                        'output': output_sentences[j],
                        'input': descriptions[k],
                        'sim_score': cosine_scores[k][j].item()  # sim score between output of j and kth input parameter of i
                    }
                    result.append(sim_dict)

        return result

    def build_edges(self, result):
        df = pd.DataFrame(result)
        df = df[df['in_id'] != df['out_id']]  # exclude connecting to oneself
        edges = df[df['sim_score'] > self.similarity_threshold]

        edges.to_csv(self.args.edge_path, index=False)

    def build_edges_by_tool(self, result):
        df = pd.DataFrame(result)
        df_filtered = df[df['in_id'] != df['out_id']]  # exclude connecting to oneself

        tool_df = pd.DataFrame(self.data)
        tool_df['tool'] = tool_df['document_content'].apply(
            lambda x: read_line(x).get('tool_name') if isinstance(x, str) else None
        )
        merged = pd.merge(df_filtered, tool_df, left_on='in_id', right_on='docid', how='left')
        merged = merged.rename(columns={'tool': 'Input_Tool'})
        merged = pd.merge(merged, tool_df, left_on='out_id', right_on='docid', how='left')
        merged = merged.rename(columns={'tool': 'Output_Tool'})

        # prune edges by threshold
        single_tool = merged[(merged['Input_Tool'] == merged['Output_Tool']) & (merged['sim_score'] > self.similarity_threshold)]
        inter_tool = merged[(merged['Input_Tool'] != merged['Output_Tool']) &
                            (merged['sim_score'] > self.similarity_threshold) &
                            (~merged['input'].str.contains('id', case=False))]

        single_tool.to_csv(self.args.single_path, index=False)
        inter_tool.to_csv(self.args.inter_path, index=False)

    def run(self):
        result = self.compute_scores()

        if self.split_by_tool:
            if not (self.args.single_path and self.args.inter_path):
                raise ValueError("--split_by_tool requires both --single_path and --inter_path")
            self.build_edges_by_tool(result)
        else:
            if not self.args.edge_path:
                raise ValueError("--edge_path is required when --split_by_tool is not set")
            self.build_edges(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="API Documentation Assistant")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSON file from build_semantic')
    parser.add_argument('--similarity_threshold', type=float, default=0.20, help='cosine similarity threshold for pruning edges')
    parser.add_argument('--split_by_tool', action='store_true', help='split edges into single-tool / inter-tool (GOATBench style)')
    # general mode output
    parser.add_argument('--edge_path', type=str, default=None, help='Path to save the pruned edge csv (used when --split_by_tool is not set)')
    # --split_by_tool mode outputs
    parser.add_argument('--single_path', type=str, default=None, help='Path to save the singletool csv file (used with --split_by_tool)')
    parser.add_argument('--inter_path', type=str, default=None, help='Path to save the intertool csv file (used with --split_by_tool)')

    args = parser.parse_args()

    query_generator = pipeline_runner(args)
    query_generator.run()
