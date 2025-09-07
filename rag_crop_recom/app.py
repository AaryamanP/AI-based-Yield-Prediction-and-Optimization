from src.rag_pipeline import RAGCropRecommender
from src.utils import parse_soil_input


def main():
    print("=== RAG-Based Crop Recommendation System ===\n")

    # # Farmer input
    # crop_name = input("Enter Crop Name (RICE, WHEAT, MAIZE, SORGHUM, PEARL MILLET, FINGER MILLET, BARLEY, CHICKPEA): ")
    # district_name = input("Enter District Name: ")
    # predicted_yield = float(input("Enter Predicted Yield (kg/ha): "))
    # soil_input = input("Enter Soil Type Percent (example: LOAMY:60, CLAY:40): ")
    # fertilizer_input = input("Enter Fertilizer Info (example: N:50,P:30,K:20): ")
    crop_name = "RICE"
    district_name = "JAIPUR"
    predicted_yield = 121.0
    soil_input = "LOAMY:60, CLAY:40"
    fertilizer_input = "N:50,P:30,K:20"





    # Parse soil + fertilizer
    soil_dict = parse_soil_input(soil_input)
    fert_dict = parse_soil_input(fertilizer_input)

    # Initialize recommender (CPU/GPU handled inside rag_pipeline)
    recommender = RAGCropRecommender(model_name="google/flan-t5-base")

    # Get recommendation
    recommendations = recommender.get_recommendation(
        crop_name=crop_name,
        district_name=district_name,
        soil_dict=soil_dict,
        fert_dict=fert_dict,
        predicted_yield=predicted_yield,
    )

    # Display results
    print("\n=== Recommended Actions ===")
    print(f"\nQuery:\n{recommendations['query']}\n")
    print(f"Retrieved Docs: {recommendations['retrieved_docs']}\n")
    print("Recommendation:\n")
    print(recommendations['recommendation'])


if __name__ == "__main__":
    main()
