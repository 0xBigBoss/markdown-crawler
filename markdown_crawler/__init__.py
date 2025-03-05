from bs4 import BeautifulSoup
import urllib.parse
import threading
from markdownify import markdownify as md
import requests
import logging
import queue
import time
import os
import re
from typing import (
    List,
    Optional,
    Union
)
__version__ = '0.1'
__author__ = 'Paul Pierre (github.com/paulpierre)'
__copyright__ = "(C) 2023 Paul Pierre. MIT License."
__contributors__ = ['Paul Pierre']

BANNER = """
                |                                     |
 __ `__ \    _` |        __|   __|   _` | \ \  \   /  |   _ \   __|
 |   |   |  (   |       (     |     (   |  \ \  \ /   |   __/  |
_|  _|  _| \__._|      \___| _|    \__._|   \_/\_/   _| \___| _|

-------------------------------------------------------------------------
A multithreaded 🕸️ web crawler that recursively crawls a website and
creates a 🔽 markdown file for each page by https://github.com/paulpierre
-------------------------------------------------------------------------
"""

logger = logging.getLogger(__name__)
DEFAULT_BASE_DIR = 'markdown'
DEFAULT_IMAGES_DIR = 'images'
DEFAULT_MAX_DEPTH = 3
DEFAULT_NUM_THREADS = 5
DEFAULT_TARGET_CONTENT = ['article', 'div', 'main', 'p']
DEFAULT_TARGET_LINKS = ['body']
DEFAULT_DOMAIN_MATCH = True
DEFAULT_BASE_PATH_MATCH = True
DEFAULT_SAVE_IMAGES = True


# --------------
# URL validation
# --------------
def is_valid_url(url: str) -> bool:
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        logger.debug(f'❌ Invalid URL {url}')
        return False


# ----------------
# Clean up the URL
# ----------------
def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), None, None, None))


# ------------------
# Save image to disk
# ------------------
def save_image(img_url: str, base_url: str, images_dir: str) -> Optional[str]:
    if not is_valid_url(img_url):
        # Try to join with base_url if it's a relative path
        img_url = urllib.parse.urljoin(base_url, img_url)
        if not is_valid_url(img_url):
            logger.debug(f'❌ Invalid image URL: {img_url}')
            return None

    try:
        # Create a filename from the URL path
        img_path = urllib.parse.urlparse(img_url).path
        img_name = img_path.split('/')[-1]

        # Remove query parameters if present
        if '?' in img_name:
            img_name = img_name.split('?')[0]

        # Ensure the name has a valid extension
        if not any(img_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']):
            img_name = f"{img_name}.jpg"  # Default to jpg if no extension

        # Create unique filename to avoid overwriting
        img_file_path = os.path.join(images_dir, img_name)
        counter = 1
        while os.path.exists(img_file_path):
            name_parts = os.path.splitext(img_name)
            img_file_path = os.path.join(images_dir, f"{name_parts[0]}_{counter}{name_parts[1]}")
            counter += 1

        # Download the image
        response = requests.get(img_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(img_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            logger.debug(f'💾 Saved image: {img_file_path}')
            return img_file_path
        else:
            logger.debug(f'❌ Failed to download image: {img_url}, status code: {response.status_code}')
            return None
    except Exception as e:
        logger.debug(f'❌ Error saving image {img_url}: {e}')
        return None


# ----------------------------
# Process images in HTML content
# ----------------------------
def process_images(soup: BeautifulSoup, base_url: str, images_dir: str) -> BeautifulSoup:
    # Create images directory if it doesn't exist
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Find all image tags
    for img in soup.find_all('img'):
        img_src = img.get('src')
        if img_src:
            local_path = save_image(img_src, base_url, images_dir)
            if local_path:
                # Update the src to point to the local file
                img['src'] = os.path.relpath(local_path, os.path.dirname(images_dir))

    # Find all picture elements with source tags
    for source in soup.find_all('source'):
        if source.get('srcset'):
            # Handle srcset attribute (might contain multiple URLs)
            srcset = source.get('srcset')
            # Simple handling - just grab the first URL before any size descriptors
            first_url = srcset.split(',')[0].strip().split(' ')[0]
            local_path = save_image(first_url, base_url, images_dir)
            if local_path:
                source['srcset'] = os.path.relpath(local_path, os.path.dirname(images_dir))

    return soup


# ------------------
# HTML parsing logic
# ------------------
def crawl(
    url: str,
    base_url: str,
    already_crawled: set,
    file_path: str,
    target_links: Union[str, List[str]] = DEFAULT_TARGET_LINKS,
    target_content: Union[str, List[str]] = None,
    valid_paths: Union[str, List[str]] = None,
    is_domain_match: Optional[bool] = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: Optional[bool] = DEFAULT_BASE_PATH_MATCH,
    is_links: Optional[bool] = False,
    is_save_images: Optional[bool] = DEFAULT_SAVE_IMAGES,
    images_dir: Optional[str] = DEFAULT_IMAGES_DIR
) -> List[str]:

    if url in already_crawled:
        return []
    try:
        logger.debug(f'Crawling: {url}')
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        logger.error(f'❌ Request error for {url}: {e}')
        return []
    if 'text/html' not in response.headers.get('Content-Type', ''):
        logger.error(f'❌ Content not text/html for {url}')
        return []
    already_crawled.add(url)

    # ---------------------------------
    # List of elements we want to strip
    # ---------------------------------
    strip_elements = []

    if is_links:
        strip_elements = ['a']

    # -------------------------------
    # Create BS4 instance for parsing
    # -------------------------------
    soup = BeautifulSoup(response.text, 'html.parser')

    # Strip unwanted tags
    for script in soup(['script', 'style']):
        script.decompose()

    # --------------------------------------------
    # Write the markdown file if it does not exist
    # --------------------------------------------
    if not os.path.exists(file_path):

        file_name = file_path.split("/")[-1]

        # ------------------
        # Get target content
        # ------------------
        content = get_target_content(soup, target_content=target_content)

        if content:
            # ----------------------------
            # Process images if requested
            # ----------------------------
            content_soup = BeautifulSoup(content, 'html.parser')

            if is_save_images:
                # Ensure images directory exists relative to the markdown file location
                page_images_dir = os.path.join(os.path.dirname(file_path), images_dir)
                if not os.path.exists(page_images_dir):
                    os.makedirs(page_images_dir)

                # Process and save images
                content_soup = process_images(content_soup, url, page_images_dir)
                content = str(content_soup)

            # --------------
            # Parse markdown
            # --------------
            output = md(
                content,
                keep_inline_images_in=['td', 'th', 'a', 'figure'],
                strip=strip_elements
            )

            logger.info(f'Created 📝 {file_name}')

            # ------------------------------
            # Write markdown content to file
            # ------------------------------
            with open(file_path, 'w') as f:
                f.write(output)
        else:
            logger.error(f'❌ Empty content for {file_path}. Please check your targets skipping.')

    child_urls = get_target_links(
        soup,
        base_url,
        target_links,
        valid_paths=valid_paths,
        is_domain_match=is_domain_match,
        is_base_path_match=is_base_path_match
    )

    logger.debug(f'Found {len(child_urls) if child_urls else 0} child URLs')
    return child_urls


def get_target_content(
    soup: BeautifulSoup,
    target_content: Union[List[str], None] = None
) -> str:

    content = ''

    # -------------------------------------
    # Get target content by target selector
    # -------------------------------------
    if target_content:
        for target in target_content:
            for tag in soup.select(target):
                content += f'{str(tag)}'.replace('\n', '')

    # ---------------------------
    # Naive estimation of content
    # ---------------------------
    else:
        max_text_length = 0
        for tag in soup.find_all(DEFAULT_TARGET_CONTENT):
            text_length = len(tag.get_text())
            if text_length > max_text_length:
                max_text_length = text_length
                main_content = tag

        content = str(main_content)

    return content if len(content) > 0 else False


def get_target_links(
    soup: BeautifulSoup,
    base_url: str,
    target_links: List[str] = DEFAULT_TARGET_LINKS,
    valid_paths: Union[List[str], None] = None,
    is_domain_match: Optional[bool] = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: Optional[bool] = DEFAULT_BASE_PATH_MATCH
) -> List[str]:

    child_urls = []

    # Get all urls from target_links
    for target in soup.find_all(target_links):
        # Get all the links in target
        for link in target.find_all('a'):
            child_urls.append(urllib.parse.urljoin(base_url, link.get('href')))

    result = []
    for u in child_urls:

        child_url = urllib.parse.urlparse(u)

        # ---------------------------------
        # Check if domain match is required
        # ---------------------------------
        if is_domain_match and child_url.netloc != urllib.parse.urlparse(base_url).netloc:
            continue

        if is_base_path_match and child_url.path.startswith(urllib.parse.urlparse(base_url).path):
            result.append(u)
            continue

        if valid_paths:
            for valid_path in valid_paths:
                if child_url.path.startswith(urllib.parse.urlparse(valid_path).path):
                    result.append(u)
                    break

    return result


# ------------------
# Worker thread logic
# ------------------
def worker(
    q: object,
    base_url: str,
    max_depth: int,
    already_crawled: set,
    base_dir: str,
    target_links: Union[List[str], None] = DEFAULT_TARGET_LINKS,
    target_content: Union[List[str], None] = None,
    valid_paths: Union[List[str], None] = None,
    is_domain_match: bool = None,
    is_base_path_match: bool = None,
    is_links: Optional[bool] = False,
    is_save_images: Optional[bool] = DEFAULT_SAVE_IMAGES,
    images_dir: Optional[str] = DEFAULT_IMAGES_DIR
) -> None:

    while not q.empty():
        depth, url = q.get()
        if depth > max_depth:
            continue
        file_name = '-'.join(re.findall(r'\w+', urllib.parse.urlparse(url).path))
        file_name = 'index' if not file_name else file_name
        file_path = f'{base_dir.rstrip("/") + "/"}{file_name}.md'

        child_urls = crawl(
            url,
            base_url,
            already_crawled,
            file_path,
            target_links,
            target_content,
            valid_paths,
            is_domain_match,
            is_base_path_match,
            is_links,
            is_save_images,
            images_dir
        )
        child_urls = [normalize_url(u) for u in child_urls]
        for child_url in child_urls:
            q.put((depth + 1, child_url))
        time.sleep(1)


# -----------------
# Thread management
# -----------------
def md_crawl(
        base_url: str,
        max_depth: Optional[int] = DEFAULT_MAX_DEPTH,
        num_threads: Optional[int] = DEFAULT_NUM_THREADS,
        base_dir: Optional[str] = DEFAULT_BASE_DIR,
        target_links: Union[str, List[str]] = DEFAULT_TARGET_LINKS,
        target_content: Union[str, List[str]] = None,
        valid_paths: Union[str, List[str]] = None,
        is_domain_match: Optional[bool] = None,
        is_base_path_match: Optional[bool] = None,
        is_debug: Optional[bool] = False,
        is_links: Optional[bool] = False,
        is_save_images: Optional[bool] = DEFAULT_SAVE_IMAGES,
        images_dir: Optional[str] = DEFAULT_IMAGES_DIR
) -> None:
    if is_domain_match is False and is_base_path_match is True:
        raise ValueError('❌ Domain match must be True if base match is set to True')

    is_domain_match = DEFAULT_DOMAIN_MATCH if is_domain_match is None else is_domain_match
    is_base_path_match = DEFAULT_BASE_PATH_MATCH if is_base_path_match is None else is_base_path_match
    is_save_images = DEFAULT_SAVE_IMAGES if is_save_images is None else is_save_images

    if not base_url:
        raise ValueError('❌ Base URL is required')

    if isinstance(target_links, str):
        target_links = target_links.split(',') if ',' in target_links else [target_links]

    if isinstance(target_content, str):
        target_content = target_content.split(',') if ',' in target_content else [target_content]

    if isinstance(valid_paths, str):
        valid_paths = valid_paths.split(',') if ',' in valid_paths else [valid_paths]

    if is_debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.debug('🐞 Debugging enabled')
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(f'🕸️ Crawling {base_url} at ⏬ depth {max_depth} with 🧵 {num_threads} threads')
    if is_save_images:
        logger.info(f'📸 Saving images to {os.path.join(base_dir, images_dir)}')

    # Validate the base URL
    if not is_valid_url(base_url):
        raise ValueError('❌ Invalid base URL')

    # Create base_dir if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create images directory if needed
    if is_save_images:
        images_path = os.path.join(base_dir, images_dir)
        if not os.path.exists(images_path):
            os.makedirs(images_path)

    already_crawled = set()

    # Create a queue of URLs to crawl
    q = queue.Queue()

    # Add the base URL to the queue
    q.put((0, base_url))

    threads = []

    # Create a thread for each URL in the queue
    for i in range(num_threads):
        t = threading.Thread(
            target=worker,
            args=(
                q,
                base_url,
                max_depth,
                already_crawled,
                base_dir,
                target_links,
                target_content,
                valid_paths,
                is_domain_match,
                is_base_path_match,
                is_links,
                is_save_images,
                images_dir
            )
        )
        threads.append(t)
        t.start()
        logger.debug(f'Started thread {i+1} of {num_threads}')

    # Wait for all threads to finish
    for t in threads:
        t.join()

    logger.info('🏁 All threads have finished')
