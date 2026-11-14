# Uso: make canvi m="tu mensaje aqui
canvi:
	@if [ -z "$(m)" ]; then \
		echo "Error: falta el mensaje. Usa: make canvi m=\"tu mensaje\""; \
		exit 1; \
	fi
	git add .
	git commit -m "$(m)"

# Uso: make puja
puja:
	git push

# Extra opcional para descargar cambios: make baixa
baixa:
	git pull