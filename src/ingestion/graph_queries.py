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

    def find_linked_contracts(self, filename: str):
        """
        Finds linked contracts (e.g., SOWs linked to an MSA).
        Note: This assumes relationships like (:Contract)-[:REFERENCES]->(:Contract) 
        are created, which we can add in the builder logic later.
        """
        query = """
        MATCH (c1:Contract {filename: $filename})-[r:REFERENCES|AMENDS|GOVERNED_BY]-(c2:Contract)
        RETURN c2.filename AS filename, type(r) AS relationship
        """
        with self.driver.session() as session:
            result = session.run(query, filename=filename)
            return [record.data() for record in result]

if __name__ == "__main__":
    queries = GraphQueries()
    print("All Contracts:")
    print(queries.get_all_contracts())
    queries.close()
