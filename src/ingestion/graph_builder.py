import os
import json
import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    Builds a Knowledge Graph in Neo4j from processed document data.
    """
    
    def __init__(self):
        Config.validate()
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def clear_database(self):
        """
        Deletes all nodes and relationships. Use with caution!
        """
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared.")

    def create_contract_node(self, filename: str, doc_type: str = "Contract"):
        """
        Creates a Contract node.
        """
        query = """
        MERGE (c:Contract {filename: $filename})
        SET c.type = $doc_type
        RETURN c
        """
        with self.driver.session() as session:
            session.run(query, filename=filename, doc_type=doc_type)

    def create_clause_nodes(self, filename: str, chunks: List[Dict[str, Any]]):
        """
        Creates Clause nodes and links them to their Contract.
        """
        query = """
        MATCH (c:Contract {filename: $filename})
        UNWIND $chunks AS chunk
        CREATE (cl:Clause {
            content: chunk.content,
            section: chunk.metadata.section,
            page_number: chunk.metadata.page_number
        })
        CREATE (c)-[:HAS_CLAUSE]->(cl)
        """
        # We use CREATE for clauses because multiple chunks might have the same content 
        # (though unlikely in this context, we want them as separate clause instances)
        with self.driver.session() as session:
            session.run(query, filename=filename, chunks=chunks)

    def create_entity_nodes(self, filename: str, entities: Dict[str, Any]):
        """
        Creates nodes for Parties, Dates, Obligations, and Amounts.
        """
        with self.driver.session() as session:
            # 1. Parties
            parties = entities.get("Parties", {})
            if isinstance(parties, dict):
                for role, name in parties.items():
                    if name:
                        session.run("""
                        MATCH (c:Contract {filename: $filename})
                        MERGE (p:Party {name: $name})
                        MERGE (c)-[:BETWEEN_PARTIES {role: $role}]->(p)
                        """, filename=filename, name=name, role=role)
            
            # 2. Dates
            dates = entities.get("Key Dates", {})
            if isinstance(dates, dict):
                for date_type, date_val in dates.items():
                    if date_val:
                        session.run("""
                        MATCH (c:Contract {filename: $filename})
                        MERGE (d:Date {value: $date_val})
                        MERGE (c)-[:HAS_DATE {type: $date_type}]->(d)
                        """, filename=filename, date_val=date_val, date_type=date_type)

            # 3. Monetary Values (Amounts)
            amounts = entities.get("Monetary Values", {})
            if isinstance(amounts, dict):
                for amt_type, amt_val in amounts.items():
                    if amt_val:
                        session.run("""
                        MATCH (c:Contract {filename: $filename})
                        MERGE (a:Amount {value: $amt_val, type: $amt_type})
                        MERGE (c)-[:HAS_VALUE]->(a)
                        """, filename=filename, amt_val=amt_val, amt_type=amt_type)

            # 4. Obligations
            obligations = entities.get("Key Obligations", [])
            if isinstance(obligations, list):
                for obl in obligations:
                    session.run("""
                    MATCH (c:Contract {filename: $filename})
                    CREATE (o:Obligation {description: $obl})
                    CREATE (c)-[:HAS_OBLIGATION]->(o)
                    """, filename=filename, obl=obl)

    def link_contracts(self):
        """
        Creates relationships between related contracts (e.g., SOW governed by MSA).
        Simple heuristic: SOWs are governed by the MSA in the same folder.
        """
        with self.driver.session() as session:
            # Link SOW to MSA
            session.run("""
            MATCH (msa:Contract), (sow:Contract)
            WHERE msa.type = 'MSA' AND sow.type = 'SOW'
            MERGE (sow)-[:GOVERNED_BY]->(msa)
            """)
            
            # Link Amendment to MSA/SOW
            session.run("""
            MATCH (amd:Contract), (target:Contract)
            WHERE amd.type = 'Amendment' AND (target.type = 'MSA' OR target.type = 'SOW')
            AND amd.filename CONTAINS target.filename
            MERGE (amd)-[:AMENDS]->(target)
            """)
            logger.info("Contract relationships linked.")

    def populate_from_directory(self, output_dir: str):
        """
        Processes all JSON files in the output directory and populates the graph.
        """
        files = [f for f in os.listdir(output_dir) if f.endswith('_processed.json')]
        logger.info(f"Found {len(files)} processed files to import into Neo4j.")
        
        for filename in files:
            file_path = os.path.join(output_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                orig_filename = data.get("filename")
                chunks = data.get("chunks", [])
                entities = data.get("entities", {})
                
                logger.info(f"Importing {orig_filename}...")
                
                # Determine doc type
                doc_type = "Contract"
                if "msa" in orig_filename.lower(): doc_type = "MSA"
                elif "sow" in orig_filename.lower(): doc_type = "SOW"
                elif "nda" in orig_filename.lower(): doc_type = "NDA"
                elif "amendment" in orig_filename.lower(): doc_type = "Amendment"
                
                self.create_contract_node(orig_filename, doc_type)
                self.create_clause_nodes(orig_filename, chunks)
                self.create_entity_nodes(orig_filename, entities)
                
                logger.info(f"Successfully imported {orig_filename}")
                
            except Exception as e:
                logger.error(f"Failed to import {filename}: {str(e)}")
        
        # Finally, create links between contracts
        self.link_contracts()

if __name__ == "__main__":
    builder = GraphBuilder()
    # builder.clear_database() # Uncomment to start fresh
    builder.populate_from_directory(Config.OUTPUT_DIR)
    builder.close()
