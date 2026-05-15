from typing import List, Dict, Any
from neo4j import GraphDatabase
from src.config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphQueries:
    """
    Helper class for executing common Cypher queries for legal intelligence.
    """
    
    def __init__(self):
        Config.validate()
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def get_all_contracts(self):
        query = "MATCH (c:Contract) RETURN c.filename AS filename, c.type AS type"
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    def get_clauses_for_contract(self, filename: str):
        query = """
        MATCH (c:Contract {filename: $filename})-[:HAS_CLAUSE]->(cl:Clause)
        RETURN cl.section AS section, cl.content AS content, cl.page_number AS page
        ORDER BY cl.page_number, cl.section
        """
        with self.driver.session() as session:
            result = session.run(query, filename=filename)
            return [record.data() for record in result]

    def get_contracts_by_party(self, party_name: str):
        query = """
        MATCH (p:Party)-[:BETWEEN_PARTIES]-(c:Contract)
        WHERE p.name CONTAINS $party_name
        RETURN c.filename AS filename, c.type AS type
        """
        with self.driver.session() as session:
            result = session.run(query, party_name=party_name)
            return [record.data() for record in result]

    def get_contract_chain(self):
        """
        Retrieves the hierarchy: MSA -> SOW -> Amendment
        """
        query = """
        MATCH (msa:Contract {type: 'MSA'})
        OPTIONAL MATCH (sow:Contract {type: 'SOW'})-[:GOVERNED_BY]->(msa)
        OPTIONAL MATCH (amd:Contract {type: 'Amendment'})-[:AMENDS]->(target)
        WHERE target = msa OR target = sow
        RETURN msa.filename AS MSA, sow.filename AS SOW, amd.filename AS Amendment
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    def expand_context(self, filenames: List[str]) -> List[Dict[str, Any]]:
        """
        For a given list of contract filenames, finds related contracts 
        (e.g. parent MSA for an SOW) and retrieves their clauses.
        """
        query = """
        MATCH (c:Contract)
        WHERE c.filename IN $filenames
        MATCH (c)-[:GOVERNED_BY|AMENDS*1..2]-(related:Contract)
        WHERE NOT related.filename IN $filenames
        MATCH (related)-[:HAS_CLAUSE]->(cl:Clause)
        RETURN related.filename AS source, cl.section AS section, cl.content AS content, cl.page_number AS page
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, filenames=filenames)
            return [record.data() for record in result]

if __name__ == "__main__":
    queries = GraphQueries()
    print("All Contracts:")
    print(queries.get_all_contracts())
    queries.close()
