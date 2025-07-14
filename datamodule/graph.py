# class GraphCfcDataset(Dataset):
#     def __init__(self,
#                  data,
#                  code_tokenizer,
#                  structure_token_id,
#                  max_seq_length=2048,
#                  max_structure_length=512):
#         super(GraphCfcDataset, self).__init__()
#         self.data = data
#         self.max_seq_length = max_seq_length
#         self.code_tokenizer = code_tokenizer
#         self.structure_token_id = structure_token_id
#         self.max_structure_length = max_structure_length

#         self.node_type_encoder = LabelEncoder().fit(NODE_TYPES)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, ind):
#         # indexing the chunked data directly
#         # source_tokens = torch.tensor(self.data[ind]['token_ids'])
#         fim_prefix, fim_suffix, fim_middle = torch.tensor([1]), torch.tensor([3]), torch.tensor([2])
#         left_context_ids = torch.tensor(self.data[ind]['lc_token_ids'])
#         right_context_ids = torch.tensor(self.data[ind]['rc_token_ids'])
#         cfc_ids = torch.tensor(self.data[ind]['cfc_token_ids'])
#         target_ids = torch.tensor(self.data[ind]['tgt_token_ids'])

#         cfc = self.code_tokenizer.decode(cfc_ids)
#         graph_data = code_to_graph(cfc)  # cfc is usually hidden in comments
#         if graph_data:
#             graph, node_features = graph_data['graph'], graph_data['node_features']
#             num_nodes = len(node_features)
#             edges_ids = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()

#             nodes_ids = torch.zeros((num_nodes), dtype=torch.long)
#             for idx, features in node_features.items():
#                 nodes_ids[idx] = self.node_type_encoder.transform([features['type']])[0]

#             num_structure_tokens = math.ceil(len(nodes_ids) / self.max_structure_length)

#         else:
#             print('Failed to extract graph from ')
#             print(cfc)
#             print('-------')
#             num_structure_tokens = 0
#             nodes_ids = None
#             edges_ids = None

#         input_ids = torch.cat([
#             fim_prefix,
#             left_context_ids,
#             fim_suffix,
#             right_context_ids,
#             torch.tensor([self.structure_token_id] * num_structure_tokens),
#             fim_middle,
#             target_ids]).to(torch.long)

#         item = {'input_ids': input_ids, 'structure_ids': nodes_ids,
#                 'edges_ids': edges_ids, 'num_structure_tokens': num_structure_tokens, 'code': cfc}
#         return item
