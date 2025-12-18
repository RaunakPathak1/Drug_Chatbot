"""
Test script to evaluate RAG system performance with 10 challenging questions
"""
import sys
from rag_pipeline.rag_pipeline import rag_answer_paracetamol, rag_answer_insulin

# 10 Test Questions
test_questions = [
    # Questions that should work well
    {
        "id": 1,
        "question": "What is Insulin Lanadelray's current role and which organization does she work for?",
        "category": "Should Work Well",
        "expected": "Senior Data Engineer at King's College Hospital NHS Foundation Trust"
    },
    {
        "id": 2,
        "question": "What are the three main hobbies or passions mentioned for Paracetamol Chad and how do they relate to his work?",
        "category": "Should Work Well",
        "expected": "Hiking (problem-solving), Photography (clarity/visualization), Cooking (iterative processes)"
    },
    {
        "id": 3,
        "question": "What programming languages did Paracetamol Chad use in his first job at StartupForge?",
        "category": "Should Work Well",
        "expected": "Python and Django"
    },
    
    # Challenging questions - edge cases
    {
        "id": 4,
        "question": "Compare the career progression timelines of both Paracetamol Chad and Insulin Lanadelray. Who advanced faster to senior roles?",
        "category": "Challenge - Cross-doc comparison",
        "expected": "Insulin advanced faster (senior role at age 25, vs Paracetamol at age 30)"
    },
    {
        "id": 5,
        "question": "What are the specific metrics or performance improvements both professionals achieved, and which one had the more significant impact?",
        "category": "Challenge - Numeric synthesis",
        "expected": "Various metrics: Chad 40% latency reduction, Insulin 40% screening acceleration, 25% cost reduction"
    },
    {
        "id": 6,
        "question": "How do Paracetamol Chad's communication methods (cooking metaphors, hiking parallels) compare with Insulin Lanadelray's communication approach (cycling analogies, data visualization)?",
        "category": "Challenge - Pattern recognition",
        "expected": "Both use hobby analogies; Chad uses cooking/hiking/photography, Insulin uses cycling/travel"
    },
    {
        "id": 7,
        "question": "Based on their hobbies and work experience, what technology stack would be ideal if these two professionals collaborated on a project together?",
        "category": "Challenge - Inference",
        "expected": "Hybrid ML/Data stack combining LLMs, data pipelines, and cloud platforms"
    },
    {
        "id": 8,
        "question": "What was the turning point or motivation mentioned for each professional to pivot their careers, and how similar were their reasons?",
        "category": "Challenge - Implicit context",
        "expected": "Chad: pandemic accelerated pivot to AI; Insulin: eager for deeper technical challenges"
    },
    {
        "id": 9,
        "question": "If Insulin Lanadelray joined Paracetamol Chad's current organization (Frontier AI Labs), what would be her most valuable contribution based on her background?",
        "category": "Challenge - Speculative synthesis",
        "expected": "Data infrastructure/pipeline expertise for LLM training and deployment"
    },
    {
        "id": 10,
        "question": "What specific challenges or pain points are implied but not explicitly stated in their career descriptions, and how did they overcome them?",
        "category": "Challenge - Implicit inference",
        "expected": "Remote work challenges, scaling data systems, ethical considerations in AI/healthcare"
    }
]

def run_tests():
    print("=" * 100)
    print("RAG SYSTEM TEST SUITE")
    print("=" * 100)
    print(f"\nTesting with {len(test_questions)} questions")
    print("Sentence chunks: 8 sentences per chunk with 30% overlap\n")
    
    results = []
    
    for test in test_questions:
        print(f"\n{'=' * 100}")
        print(f"TEST #{test['id']}: {test['category']}")
        print(f"{'=' * 100}")
        print(f"\nQuestion: {test['question']}\n")
        print(f"Expected Context: {test['expected']}\n")
        
        try:
            # Route to appropriate agent based on keywords
            if 'paracetamol' in test['question'].lower():
                answer = rag_answer_paracetamol(test['question'])
            elif 'insulin' in test['question'].lower():
                answer = rag_answer_insulin(test['question'])
            else:
                # Try both if not specified
                print("Question mentions both or neither - testing with Insulin first...")
                answer = rag_answer_insulin(test['question'])
            
            print(f"RAG Answer: {answer}\n")
            
            results.append({
                "id": test['id'],
                "question": test['question'],
                "category": test['category'],
                "answer": answer,
                "success": answer and "don't know" not in answer.lower()
            })
            
        except Exception as e:
            print(f"ERROR: {str(e)}\n")
            results.append({
                "id": test['id'],
                "question": test['question'],
                "category": test['category'],
                "answer": f"ERROR: {str(e)}",
                "success": False
            })
    
    # Summary Report
    print(f"\n\n{'=' * 100}")
    print("SUMMARY REPORT")
    print(f"{'=' * 100}\n")
    
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    
    print(f"Total Questions: {len(results)}")
    print(f"Successful Answers: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"Failed/Inconclusive: {failed} ({failed/len(results)*100:.1f}%)\n")
    
    print("BREAKDOWN BY CATEGORY:")
    print("-" * 100)
    
    should_work = [r for r in results if "Should Work Well" in r['category']]
    challenges = [r for r in results if "Challenge" in r['category']]
    
    should_work_success = sum(1 for r in should_work if r['success'])
    challenge_success = sum(1 for r in challenges if r['success'])
    
    print(f"\n'Should Work Well' Category: {should_work_success}/{len(should_work)} successful")
    for r in should_work:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"  Q{r['id']}: {status}")
    
    print(f"\n'Challenge' Category: {challenge_success}/{len(challenges)} successful")
    for r in challenges:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"  Q{r['id']}: {status} - {r['category']}")
    
    print(f"\n{'=' * 100}\n")

if __name__ == "__main__":
    run_tests()
