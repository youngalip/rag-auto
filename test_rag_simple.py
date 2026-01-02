from rag_answer_fixed import RAGAnswerGenerator
import time

# =====================================================
# SIMPLE RAG TESTING SCRIPT
# =====================================================

def test_rag():
    """
    Test RAG model dengan berbagai query
    """
    
    # Test queries covering different topics
    test_queries = [
        "bagaimana cara mengajukan bebas lab?",
        "prosedur pengajuan cuti kuliah",
        "cara bayar ukt",
        "syarat yudisium",
        "dimana saya bisa melihat jadwal kuliah?",
        "bagaimana cara login sinau digital?",
        "apa itu melisa?",
        "bagaimana cara membayar ukt dengan virtual account?",
        "apa saja persyaratan untuk mengikuti tes tep?",
        "dimana lokasi perpustakaan?"
    ]
    
    print("="*60)
    print("🧪 TESTING RAG MODEL")
    print("="*60)
    print(f"\nTotal test queries: {len(test_queries)}")
    print("\n" + "="*60)
    
    # Initialize generator
    print("\nInitializing RAG Generator...")
    generator = RAGAnswerGenerator()
    print("✓ Generator ready")
    
    # Run tests
    results = []
    total_time = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_queries)}")
        print('='*60)
        print(f"Query: {query}")
        print("-"*60)
        
        # Generate answer
        start_time = time.time()
        result = generator.generate(query, verbose=False)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Display result
        answer = result['answer']
        word_count = len(answer.split())
        
        print(f"Answer ({word_count} words, {elapsed:.2f}s):")
        print(answer)
        
        # Check for warnings
        if result.get('warning'):
            print(f"\n⚠️ Warning: {result['warning']}")
        
        # Check if low confidence
        if result.get('retrieved_docs') and result['retrieved_docs']:
            if result['retrieved_docs'][0].get('low_confidence'):
                print("⚠️ Low confidence retrieval")
        
        # Store result
        results.append({
            'query': query,
            'answer': answer,
            'word_count': word_count,
            'time': elapsed,
            'warning': result.get('warning')
        })
        
        print("="*60)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    avg_words = sum(r['word_count'] for r in results) / len(results)
    avg_time = total_time / len(results)
    
    print(f"\nTotal queries tested: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per query: {avg_time:.2f}s")
    print(f"Average answer length: {avg_words:.1f} words")
    
    # Word count distribution
    short_answers = sum(1 for r in results if r['word_count'] < 20)
    medium_answers = sum(1 for r in results if 20 <= r['word_count'] < 50)
    long_answers = sum(1 for r in results if r['word_count'] >= 50)
    
    print(f"\nAnswer length distribution:")
    print(f"  Short (<20 words): {short_answers}")
    print(f"  Medium (20-49 words): {medium_answers}")
    print(f"  Long (≥50 words): {long_answers}")
    
    # Warnings
    warnings = sum(1 for r in results if r['warning'])
    if warnings > 0:
        print(f"\n⚠️ Queries with warnings: {warnings}")
    
    print("\n" + "="*60)
    print("✅ Testing completed!")
    print("="*60)
    
    return results

def interactive_test():
    """
    Interactive testing mode
    """
    print("="*60)
    print("🤖 INTERACTIVE RAG TESTING")
    print("="*60)
    print("\nType your questions (or 'quit' to exit)")
    print("-"*60)
    
    generator = RAGAnswerGenerator()
    
    while True:
        print("\n" + "="*60)
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            print("⚠️ Please enter a question")
            continue
        
        print("-"*60)
        print("Generating answer...")
        
        result = generator.generate(query, verbose=True)
        
        print("\n" + "="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_test()
    else:
        test_rag()