"""Enrichment engine for auto-enrichment from external sources."""
import logging
from typing import List, Dict, Any, Optional
import asyncio
import httpx
import json

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False

from langchain.schema import Document
from app.services.vector_store import vector_store_service

logger = logging.getLogger(__name__)

# Trusted external sources configuration
TRUSTED_SOURCES = {
    'wikipedia': {
        'enabled': WIKIPEDIA_AVAILABLE,
        'name': 'Wikipedia',
        'description': 'General knowledge encyclopedia',
        'priority': 1
    },
    'arxiv': {
        'enabled': True,
        'name': 'arXiv',
        'description': 'Academic papers and research',
        'api_url': 'http://export.arxiv.org/api/query',
        'priority': 2
    },
    'pubmed': {
        'enabled': True,
        'name': 'PubMed',
        'description': 'Medical and health research',
        'api_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
        'priority': 3
    },
    'web_search': {
        'enabled': DUCKDUCKGO_AVAILABLE,
        'name': 'Web Search',
        'description': 'General web search results',
        'priority': 4
    }
}


class EnrichmentEngine:
    """Handles auto-enrichment from trusted external sources."""

    def __init__(self):
        self.wikipedia_available = WIKIPEDIA_AVAILABLE
        self.duckduckgo_available = DUCKDUCKGO_AVAILABLE
        self.trusted_sources = TRUSTED_SOURCES
        self.http_client = None
    
    async def auto_enrich(
        self,
        query: str,
        missing_info: List[str],
        max_sources: int = 3,
        source_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Attempt to automatically enrich knowledge base from trusted external sources.

        Args:
            query: Original user query
            missing_info: List of missing information items
            max_sources: Maximum number of external sources to fetch
            source_types: Specific source types to use (e.g., ['wikipedia', 'arxiv'])

        Returns:
            Dictionary with enrichment results
        """
        enrichment_results = {
            'sources_added': [],
            'content_added': [],
            'success': False
        }

        if not missing_info:
            return enrichment_results

        # Determine which sources to use
        if source_types is None:
            # Use all available sources in priority order
            source_types = [
                name for name, config in sorted(
                    self.trusted_sources.items(),
                    key=lambda x: x[1]['priority']
                ) if config['enabled']
            ]

        # Try each source type until we have enough results
        for source_type in source_types:
            if len(enrichment_results['sources_added']) >= max_sources:
                break

            remaining = max_sources - len(enrichment_results['sources_added'])

            try:
                if source_type == 'wikipedia' and self.wikipedia_available:
                    results = await self._enrich_from_wikipedia(query, missing_info, remaining)
                elif source_type == 'arxiv':
                    results = await self._enrich_from_arxiv(query, missing_info, remaining)
                elif source_type == 'pubmed':
                    results = await self._enrich_from_pubmed(query, missing_info, remaining)
                elif source_type == 'web_search' and self.duckduckgo_available:
                    results = await self._enrich_from_web_search(query, missing_info, remaining)
                else:
                    continue

                enrichment_results['sources_added'].extend(results['sources'])
                enrichment_results['content_added'].extend(results['content'])

            except Exception as e:
                logger.error(f"Error enriching from {source_type}: {e}")
                continue

        # Add enriched content to vector store
        if enrichment_results['content_added']:
            try:
                await self._add_enriched_content_to_store(
                    enrichment_results['content_added']
                )
                enrichment_results['success'] = True
                logger.info(f"Successfully enriched knowledge base with {len(enrichment_results['content_added'])} items from trusted sources")
            except Exception as e:
                logger.error(f"Error adding enriched content to vector store: {e}")

        return enrichment_results
    
    async def _enrich_from_wikipedia(
        self,
        query: str,
        missing_info: List[str],
        max_sources: int
    ) -> Dict[str, List]:
        """Fetch information from Wikipedia."""
        results = {'sources': [], 'content': []}
        
        if not self.wikipedia_available:
            return results
        
        try:
            # Search for relevant Wikipedia pages
            search_terms = [query] + missing_info[:2]
            
            for term in search_terms[:max_sources]:
                try:
                    # Search Wikipedia
                    search_results = wikipedia.search(term, results=1)
                    
                    if search_results:
                        page_title = search_results[0]
                        page = wikipedia.page(page_title, auto_suggest=False)
                        
                        # Get summary (first few paragraphs)
                        content = page.summary
                        
                        results['sources'].append(f"Wikipedia: {page.title}")
                        results['content'].append({
                            'text': content,
                            'source': 'wikipedia',
                            'title': page.title,
                            'url': page.url,
                            'query': term
                        })
                        
                        logger.info(f"Fetched Wikipedia page: {page.title}")
                        
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    if e.options:
                        try:
                            page = wikipedia.page(e.options[0], auto_suggest=False)
                            content = page.summary
                            
                            results['sources'].append(f"Wikipedia: {page.title}")
                            results['content'].append({
                                'text': content,
                                'source': 'wikipedia',
                                'title': page.title,
                                'url': page.url,
                                'query': term
                            })
                        except:
                            continue
                except wikipedia.exceptions.PageError:
                    logger.warning(f"Wikipedia page not found for: {term}")
                    continue
                except Exception as e:
                    logger.error(f"Error fetching from Wikipedia: {e}")
                    continue
                
                # Small delay to be respectful
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Error in Wikipedia enrichment: {e}")
        
        return results

    async def _enrich_from_arxiv(
        self,
        query: str,
        missing_info: List[str],
        max_sources: int
    ) -> Dict[str, List]:
        """Fetch information from arXiv (academic papers)."""
        results = {'sources': [], 'content': []}

        try:
            # Create HTTP client if not exists
            if self.http_client is None:
                self.http_client = httpx.AsyncClient(timeout=30.0)

            # Build search query
            search_terms = f"{query} {' '.join(missing_info[:2])}"

            # Query arXiv API
            params = {
                'search_query': f'all:{search_terms}',
                'start': 0,
                'max_results': max_sources,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }

            response = await self.http_client.get(
                self.trusted_sources['arxiv']['api_url'],
                params=params
            )

            if response.status_code == 200:
                # Parse XML response
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.text)

                # Define namespace
                ns = {'atom': 'http://www.w3.org/2005/Atom'}

                for entry in root.findall('atom:entry', ns)[:max_sources]:
                    title = entry.find('atom:title', ns).text.strip()
                    summary = entry.find('atom:summary', ns).text.strip()
                    link = entry.find('atom:id', ns).text.strip()

                    # Get authors
                    authors = [
                        author.find('atom:name', ns).text
                        for author in entry.findall('atom:author', ns)
                    ]
                    authors_str = ', '.join(authors[:3])
                    if len(authors) > 3:
                        authors_str += ' et al.'

                    results['sources'].append(f"arXiv: {title}")
                    results['content'].append({
                        'text': f"{title}\n\nAuthors: {authors_str}\n\n{summary}",
                        'source': 'arxiv',
                        'title': title,
                        'url': link,
                        'query': search_terms,
                        'authors': authors_str
                    })

                    logger.info(f"Fetched arXiv paper: {title}")

                await asyncio.sleep(0.5)  # Be respectful to API

        except Exception as e:
            logger.error(f"Error in arXiv enrichment: {e}")

        return results

    async def _enrich_from_pubmed(
        self,
        query: str,
        missing_info: List[str],
        max_sources: int
    ) -> Dict[str, List]:
        """Fetch information from PubMed (medical research)."""
        results = {'sources': [], 'content': []}

        try:
            # Create HTTP client if not exists
            if self.http_client is None:
                self.http_client = httpx.AsyncClient(timeout=30.0)

            # Build search query
            search_terms = f"{query} {' '.join(missing_info[:2])}"

            # Step 1: Search for article IDs
            search_url = f"{self.trusted_sources['pubmed']['api_url']}/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': search_terms,
                'retmax': max_sources,
                'retmode': 'json',
                'sort': 'relevance'
            }

            search_response = await self.http_client.get(search_url, params=search_params)

            if search_response.status_code == 200:
                search_data = search_response.json()
                id_list = search_data.get('esearchresult', {}).get('idlist', [])

                if id_list:
                    # Step 2: Fetch article summaries
                    summary_url = f"{self.trusted_sources['pubmed']['api_url']}/esummary.fcgi"
                    summary_params = {
                        'db': 'pubmed',
                        'id': ','.join(id_list),
                        'retmode': 'json'
                    }

                    summary_response = await self.http_client.get(summary_url, params=summary_params)

                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()

                        for pmid in id_list:
                            article = summary_data.get('result', {}).get(pmid, {})

                            if article:
                                title = article.get('title', 'Unknown')
                                authors = article.get('authors', [])
                                authors_str = ', '.join([a.get('name', '') for a in authors[:3]])
                                if len(authors) > 3:
                                    authors_str += ' et al.'

                                # Get abstract (requires another API call)
                                abstract_url = f"{self.trusted_sources['pubmed']['api_url']}/efetch.fcgi"
                                abstract_params = {
                                    'db': 'pubmed',
                                    'id': pmid,
                                    'retmode': 'xml'
                                }

                                try:
                                    abstract_response = await self.http_client.get(abstract_url, params=abstract_params)
                                    if abstract_response.status_code == 200:
                                        import xml.etree.ElementTree as ET
                                        root = ET.fromstring(abstract_response.text)
                                        abstract_elem = root.find('.//AbstractText')
                                        abstract = abstract_elem.text if abstract_elem is not None else article.get('source', 'No abstract available')
                                    else:
                                        abstract = article.get('source', 'No abstract available')
                                except:
                                    abstract = article.get('source', 'No abstract available')

                                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                                results['sources'].append(f"PubMed: {title}")
                                results['content'].append({
                                    'text': f"{title}\n\nAuthors: {authors_str}\n\n{abstract}",
                                    'source': 'pubmed',
                                    'title': title,
                                    'url': pubmed_url,
                                    'query': search_terms,
                                    'authors': authors_str,
                                    'pmid': pmid
                                })

                                logger.info(f"Fetched PubMed article: {title}")

                await asyncio.sleep(0.5)  # Be respectful to API

        except Exception as e:
            logger.error(f"Error in PubMed enrichment: {e}")

        return results

    async def _enrich_from_web_search(
        self,
        query: str,
        missing_info: List[str],
        max_sources: int
    ) -> Dict[str, List]:
        """Fetch information from web search."""
        results = {'sources': [], 'content': []}
        
        if not self.duckduckgo_available:
            return results
        
        try:
            search_term = f"{query} {' '.join(missing_info[:2])}"
            
            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    search_term,
                    max_results=max_sources
                ))
                
                for result in search_results:
                    results['sources'].append(f"Web: {result.get('title', 'Unknown')}")
                    results['content'].append({
                        'text': result.get('body', ''),
                        'source': 'web_search',
                        'title': result.get('title', 'Unknown'),
                        'url': result.get('href', ''),
                        'query': search_term
                    })
                    
                    logger.info(f"Fetched web result: {result.get('title', 'Unknown')}")
        
        except Exception as e:
            logger.error(f"Error in web search enrichment: {e}")
        
        return results
    
    async def _add_enriched_content_to_store(self, content_items: List[Dict[str, Any]]):
        """Add enriched content to the vector store."""
        documents = []
        
        for item in content_items:
            doc = Document(
                page_content=item['text'],
                metadata={
                    'document_id': f"enriched_{item['source']}_{hash(item['title'])}",
                    'filename': f"{item['source']}: {item['title']}",
                    'source': item['source'],
                    'title': item['title'],
                    'url': item.get('url', ''),
                    'enriched': True,
                    'original_query': item.get('query', '')
                }
            )
            documents.append(doc)
        
        if documents:
            await vector_store_service.add_documents(documents)
            logger.info(f"Added {len(documents)} enriched documents to vector store")
    
    def get_enrichment_capabilities(self) -> Dict[str, Any]:
        """Get available enrichment capabilities and trusted sources."""
        capabilities = {
            'auto_enrichment_enabled': any(
                source['enabled'] for source in self.trusted_sources.values()
            ),
            'trusted_sources': {}
        }

        for source_name, source_config in self.trusted_sources.items():
            capabilities['trusted_sources'][source_name] = {
                'enabled': source_config['enabled'],
                'name': source_config['name'],
                'description': source_config['description'],
                'priority': source_config['priority']
            }

        return capabilities

    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()


# Global instance
enrichment_engine = EnrichmentEngine()

