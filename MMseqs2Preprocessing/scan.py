import json



if __name__ == "__main__":
    with open("./outputs/clusters.json", "r") as file:
        data = json.load(file)

        items = []
        clusters = 0
        citems = 0
        for cluster in data:
            clusters = clusters + 1
            for item in cluster["items"]:
                citems = citems + 1
                if not item["fasta_id"] in items:
                    items.append(item["fasta_id"])
                else:
                    print(f"Item alredy exists: {item["fasta_id"]} | {item["tag"]}")

        print(f"Clusters checked : {clusters} Items checked : {citems}")